import torch
from torch import nn
from torch.nn import functional as F

from ..registry import READERS
import pdb


@READERS.register_module
class VoxelFeatureExtractorV3(nn.Module):
    def __init__(
        self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV3"
    ):
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        assert self.num_input_features == features.shape[-1]
        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)
        return points_mean.contiguous()

@READERS.register_module
class VoxelFeatureExtractorV4(nn.Module):
    def __init__(
        self, num_input_features=4, max_points_in_voxel=10, out_channels=16,
        norm_cfg=None, name="VoxelFeatureExtractorV4"
    ):
        super(VoxelFeatureExtractorV4, self).__init__()
        self.name = name
        self.num_input_features = num_input_features
        self.max_points_in_voxel = max_points_in_voxel

        self.norm_cfg = norm_cfg
        self.in_channels = max_points_in_voxel * num_input_features
        self.linear1 = nn.Linear(self.in_channels, out_channels * 2, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels * 2) #, eps=1e-3, momentum=0.01)
        self.linear2 = nn.Linear(out_channels * 2, out_channels,bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels) #, eps=1e-3, momentum=0.01)

    def forward(self, features, num_voxels, coors=None):
        if self.num_input_features == features.shape[-1] + 1:
            num_voxels_features = torch.cat(
                self.max_points_in_voxel * [num_voxels.unsqueeze(-1).unsqueeze(-1)],
                dim=1).type(torch.float)
            features = torch.cat([features, num_voxels_features], dim=-1)
        else:
            assert self.num_input_features == features.shape[-1]
        x = self.linear1(features.view(-1, self.in_channels))
        x = self.norm1(x)
        x = F.leaky_relu(x)

        x = self.linear2(x)
        x = self.norm2(x)
        x = F.leaky_relu(x)
        return x.contiguous()