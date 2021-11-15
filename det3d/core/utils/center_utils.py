# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou and Tianwei Yin 
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
from torch import nn
from .circle_nms_jit import circle_nms
import pdb
import scipy


def gaussian_radius(det_size, min_overlap=0.5): # min overlap = 0.1
    height, width = det_size
    a1  = 1
    b1  = (height + width)
    c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1  = (b1 + sq1) / 2

    a2  = 4
    b2  = 2 * (height + width)
    c2  = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2  = (b2 + sq2) / 2

    a3  = 4 * min_overlap
    b3  = -2 * min_overlap * (height + width)
    c3  = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3  = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian_radius2D(det_size):
    l, w = det_size
    return {"half_height": w // 2, "half_width": l // 2}

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def gaussian2D_rotate(shape, rotate, sigma=(1, 1)):
    m, n = [(ss - 1) // 2 for ss in shape]
    y, x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp(-(x * x) / \
               (2 * sigma[0] * sigma[0]) + -(y * y) / \
               (2 * sigma[1] * sigma[1]))
    pad_max = np.ceil(np.sqrt((m + 1) ** 2 + (n + 1) ** 2)).astype(np.int)
    h_pad = np.pad(h,
                   ((pad_max - m - 1, pad_max - m - 1),
                    (pad_max - n - 1, pad_max - n - 1)),
                   'constant', constant_values=(0, 0))
    h_pad_rotate = scipy.ndimage.rotate(
        h_pad, angle=rotate/np.pi*180, reshape=False)
    np.clip(h_pad_rotate, 0.0, 1.0, out=h_pad_rotate)
    h_pad_rotate[
        h_pad_rotate < np.finfo(h_pad_rotate.dtype).eps *
        h_pad_rotate.max()] = 0
    return h_pad_rotate, pad_max - 1

def draw_umich_gaussian2D(heatmap, center, radius, rotate=0.0, sigma_decay=3.0, k=1):
    assert isinstance(radius, dict)
    radius = np.array([radius["half_height"], radius["half_width"]])
    diameter = 2 * radius + 1 # make sure odd val.
    gaussian, radius_max = gaussian2D_rotate(
        (diameter[1], diameter[0]), rotate, sigma=diameter / sigma_decay)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius_max), min(width - x, radius_max + 1)
    top, bottom = min(y, radius_max), min(height - y, radius_max + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[
        radius_max - top:radius_max + bottom,
        radius_max - left:radius_max + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_umich_gaussian(heatmap, center, radius, rotate=0.0, k=1):
    diameter = 2 * radius + 1
    # crop rect valid shape.
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]
    # from center extend dist to left, to right, to top, to bottom.
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap  = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def _gather_feat(feat, ind, mask=None):
    # feat: N * (w * h) * 8(reg) / 1(hm)
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    # because of ind has lots of zeros because total 500 bboxes, but valid ones are gt num
    # after gather, feat will be fill by heatmap / reg vals from gt center in feature map
    # other parts will auto fill by ind 0's val.
    feat = feat.gather(1, ind)
    if mask is not None: # is None
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def _circle_nms(boxes, min_radius, post_max_size=83):
    """
    NMS according to center distance
    """
    keep = np.array(circle_nms(boxes.cpu().numpy(), thresh=min_radius))[:post_max_size]

    keep = torch.from_numpy(keep).long().to(boxes.device)

    return keep 


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)
    Returns:
    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans
