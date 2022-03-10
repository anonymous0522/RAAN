import time
import numpy as np
import math

import torch
import pdb
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer

# attention_conv2d
from biattention_conv2d_concat_initW import AttentionConv2D, AttentionConvTranspose2D


@NECKS.register_module
class RPN(nn.Module):
    def __init__(
        self,
        layer_nums,
        ds_layer_strides,
        ds_num_filters,
        us_layer_strides,
        us_num_filters,
        num_input_features,
        norm_cfg=None,
        name="rpn",
        logger=None,
        conv_attention_flag=False,
        deconv_attention_flag=False,
        **kwargs
    ):
        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides
        self._num_filters = ds_num_filters
        self._layer_nums = layer_nums
        self._upsample_strides = us_layer_strides
        self._num_upsample_filters = us_num_filters
        self._num_input_features = num_input_features
        self._conv_attention_flag = conv_attention_flag
        self._deconv_attention_flag = deconv_attention_flag
        if conv_attention_flag:
            self.conv = AttentionConv2D
        else:
            self.conv = nn.Conv2d

        if deconv_attention_flag:
            self.deconv = AttentionConvTranspose2D # AttentionConvTranspose2D, AttentionConvTransposeOri2D, AttentionConvUpSample2D
        else:
            self.deconv = nn.ConvTranspose2d
        print("self.conv ", self.conv, " self.deconv ", self.deconv)
        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            # print(upsample_strides[i])
            must_equal_list.append(
                self._upsample_strides[i]
                / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1])
            )

        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [self._num_input_features, *self._num_filters[:-1]]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(
                in_filters[i],
                self._num_filters[i],
                layer_num,
                stride=self._layer_strides[i],
            )
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                stride = (self._upsample_strides[i - self._upsample_start_idx])
                if stride > 1:
                    deblock = Sequential(
                        self.deconv(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride,
                            stride=stride,
                            bias=False))
                    if not deconv_attention_flag:
                        deblock.add(build_norm_layer(
                                self._norm_cfg,
                                self._num_upsample_filters[i - self._upsample_start_idx],
                            )[1])
                    deblock.add(nn.ReLU())
                else:
                    stride = np.round(1 / stride).astype(np.int64)
                    # kernel_size = 3 if conv_attention_flag else stride
                    # padding_size = 1 if conv_attention_flag else 0
                    deblock = Sequential(
                        self.conv(
                            num_out_filters,
                            self._num_upsample_filters[i - self._upsample_start_idx],
                            stride, # kernel_size
                            stride=stride,
                            bias=False,
                            # padding=padding_size,
                        ))
                    if not conv_attention_flag:
                        deblock.add(build_norm_layer(
                                self._norm_cfg,
                                self._num_upsample_filters[i - self._upsample_start_idx],
                            )[1])
                    deblock.add(nn.ReLU())
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        # self.init_weights()
        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)
        if len(self._upsample_strides) > 0:
            factor /= self._upsample_strides[-1]
        return factor

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):

        block = Sequential(
            self.conv(inplanes, planes, 3, padding=1, stride=stride, bias=False),
        )
        if not self._conv_attention_flag:
            block.add(build_norm_layer(self._norm_cfg, planes)[1])
        block.add(nn.ReLU())

        for j in range(num_blocks):
            block.add(self.conv(planes, planes, 3, padding=1, bias=False))
            if not self._conv_attention_flag:
                block.add(build_norm_layer(self._norm_cfg, planes)[1])
            block.add(nn.ReLU())
        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")
    '''
    def init_weights(self):
        for m in self.blocks.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        for m in self.deblocks.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        # init AttentionConv2D at the end.
        for m in self.blocks.modules():
            if isinstance(m, AttentionConv2D):
                m.init_attention_weight()
            if isinstance(m, AttentionConvTranspose2D):
                m.init_attention_weight()
        for m in self.deblocks.modules():
            if isinstance(m, AttentionConv2D):
                m.init_attention_weight()
            if isinstance(m, AttentionConvTranspose2D):
                m.init_attention_weight()
    '''

    def forward(self, x):
        ups = []
        # x: 4*256*180*180 -> 1*1 same shape
        # downsample once 90*90 -> upsample 
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i - self._upsample_start_idx >= 0:
                ups.append(self.deblocks[i - self._upsample_start_idx](x))
        if len(ups) > 0:
            x = torch.cat(ups, dim=1) # [4, 512, 180, 180]
        return x