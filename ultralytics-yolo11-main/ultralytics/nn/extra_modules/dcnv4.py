"""Flash Intern Image
A Pytorch Implementation of Flash Intern Image as decribed in:
`InternImage: Exploring Large-Scale Vision Foundation Models with Deformable Convolutions`
    - https://arxiv.org/pdf/2211.05778
`DCNv4`
    - https://arxiv.org/pdf/2401.06197
Code/weights from https://github.com/OpenGVLab/DCNv4, original copyright/license info below
"""
# --------------------------------------------------------
# Flash Intern Image
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_, constant_

# from collections import OrderedDict
# import torch.utils.checkpoint as checkpoint
# from timm.models.layers import trunc_normal_, DropPath
# from timm.layers import SelectAdaptivePool2d
# from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
# from ._registry import register_model, generate_default_cfgs
# from ._builder import build_model_with_cfg
# from ._manipulate import checkpoint_seq

import torch.nn.functional as F
from typing import Dict, Any, Tuple, Optional, List
import warnings
import logging
import math

# from ultralytics.nn.extra_modules.DCNv4_op.DCNv4 import DCNv4
# /nn/extra_modules/DCNv4_op/DCNv4/modules/dcnv4.py
from ultralytics.nn.extra_modules.DCNv4_op.DCNv4.modules.dcnv4 import DCNv4

__all__ = ['DCNv4_pytorch', 'DCNv4Block', 'CKDCNv4', 'DCNV4_Conv']


_logger = logging.getLogger(__name__)
torch.fx.wrap('len')

dcn_version = 'CUDA'
try:
    import DCNv4
except ImportError:
    dcn_version = 'pytorch'

has_yacs = True
try:
    import yacs
except ImportError:
    has_yacs = False


class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


def build_norm_layer(
        dim,
        norm_layer,
        in_format='channels_last',
        out_format='channels_last',
        eps=1e-6
    ):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')

    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')


def _get_reference_points(
        spatial_shapes: List[int],
        device: Optional[torch.device], 
        kernel_h: int, 
        kernel_w: int, 
        dilation_h: int, 
        dilation_w: int, 
        pad_h: int=0, 
        pad_w: int=0, 
        stride_h: int=1, 
        stride_w: int=1
    ):
    _, H_, W_, _ = spatial_shapes
    H_out = (H_ - (dilation_h * (kernel_h - 1) + 1)) // stride_h + 1
    W_out = (W_ - (dilation_w * (kernel_w - 1) + 1)) // stride_w + 1

    ref_y, ref_x = torch.meshgrid(
        torch.linspace(
            (dilation_h * (kernel_h - 1)) // 2 + 0.5,
            (dilation_h * (kernel_h - 1)) // 2 + 0.5 + (H_out - 1) * stride_h,
            H_out,
            dtype=torch.float32,
            device=device
        ),
        torch.linspace(
            (dilation_w * (kernel_w - 1)) // 2 + 0.5,
            (dilation_w * (kernel_w - 1)) // 2 + 0.5 + (W_out - 1) * stride_w,
            W_out,
            dtype=torch.float32,
            device=device
        )
    )
    ref_y = ref_y.reshape(-1)[None] / H_
    ref_x = ref_x.reshape(-1)[None] / W_

    ref = torch.stack((ref_x, ref_y), -1).reshape(1, H_out, W_out, 1, 2)

    return ref


def _generate_dilation_grids(
        spatial_shapes: List[int], 
        kernel_h: int, 
        kernel_w: int, 
        dilation_h: int, 
        dilation_w: int, 
        group: int, 
        device: Optional[torch.device],
    ):
    _, H_, W_, _ = spatial_shapes
    points_list = []
    x, y = torch.meshgrid(
        torch.linspace(
            -((dilation_w * (kernel_w - 1)) // 2),
            -((dilation_w * (kernel_w - 1)) // 2) + (kernel_w - 1) * dilation_w, 
            kernel_w,
            dtype=torch.float32,
            device=device
        ),
        torch.linspace(
            -((dilation_h * (kernel_h - 1)) // 2),
            -((dilation_h * (kernel_h - 1)) // 2) + (kernel_h - 1) * dilation_h, 
            kernel_h,
            dtype=torch.float32,
            device=device
        )
    )

    points_list.extend([x / W_, y / H_])
    grid = torch.stack(points_list, -1).reshape(-1, 1, 2).\
        repeat(1, group, 1).permute(1, 0, 2)
    grid = grid.reshape(1, 1, 1, group * kernel_h * kernel_w, 2)

    return grid


def dcnv4_core_pytorch(
        input, 
        offset,
        mask, 
        kernel_h: int,
        kernel_w: int, 
        stride_h: int, 
        stride_w: int, 
        pad_h: int,
        pad_w: int, 
        dilation_h: int, 
        dilation_w: int, 
        group: int,
        group_channels: int, 
        offset_scale: float
    ):
    input = F.pad(
        input,
        [0, 0, pad_h, pad_h, pad_w, pad_w]
    )
    N_, H_in, W_in, _ = input.shape
    _, H_out, W_out, _ = offset.shape

    ref = _get_reference_points(
        input.shape, 
        input.device, 
        kernel_h, 
        kernel_w, 
        dilation_h, 
        dilation_w, 
        pad_h, 
        pad_w, 
        stride_h, 
        stride_w
    )
    grid = _generate_dilation_grids(
        input.shape, 
        kernel_h, 
        kernel_w, 
        dilation_h, 
        dilation_w, 
        group, 
        input.device
    )
    spatial_norm = torch.tensor([W_in, H_in]).reshape(1, 1, 1, 2).\
        repeat(1, 1, 1, group * kernel_h * kernel_w).to(input.device)

    sampling_locations = (ref + grid * offset_scale).repeat(N_, 1, 1, 1, 1).flatten(3, 4) + \
        offset * offset_scale / spatial_norm

    P_ = kernel_h * kernel_w
    sampling_grids = 2 * sampling_locations - 1
    input_ = input.view(N_, H_in * W_in, group * group_channels).transpose(1, 2).\
        reshape(N_ * group, group_channels, H_in, W_in)
    sampling_grid_ = sampling_grids.view(N_, H_out * W_out, group, P_, 2).transpose(1, 2).\
        flatten(0, 1)
    sampling_input_ = F.grid_sample(
        input_, 
        sampling_grid_, 
        mode='bilinear', 
        padding_mode='zeros', 
        align_corners=False
    )

    mask = mask.view(N_, H_out * W_out, group, P_).transpose(1, 2).\
        reshape(N_ * group, 1, H_out * W_out, P_)
    output = (sampling_input_ * mask).sum(-1).\
        view(N_, group * group_channels, H_out * W_out)

    return output.transpose(1, 2).reshape(N_, H_out, W_out, -1).contiguous()


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError(
            "invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))

    return (n & (n - 1) == 0) and n != 0


class CenterFeatureScaleModule(nn.Module):
    def forward(
            self,
            query,
            center_feature_scale_proj_weight,
            center_feature_scale_proj_bias
        ):
        center_feature_scale = \
            F.linear(
                query,
                weight=center_feature_scale_proj_weight,
                bias=center_feature_scale_proj_bias
            ).sigmoid()
        return center_feature_scale


class DCNv4_pytorch(nn.Module):
    def __init__(
            self,
            channels=128,
            kernel_size=3,
            stride=1,
            pad=1,
            dilation=1,
            group=4,
            offset_scale=1.0,
            dw_kernel_size=3,
            remove_center=False,
            output_bias=True,
            without_pointwise=False,
            **kwargs
        ):
        super().__init__()
        if channels % group != 0:
            raise ValueError(
                f'channels must be divisible by group, but got {channels} and {group}')
        _d_per_group = channels // group

        # you'd better set _d_per_group to a power of 2 which is more efficient in our CUDA implementation
        assert _d_per_group % 16 == 0

        self.offset_scale = offset_scale
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.pad = pad
        self.group = group
        self.group_channels = channels // group
        self.offset_scale = offset_scale
        self.dw_kernel_size = dw_kernel_size
        self.remove_center = int(remove_center)
        self.without_pointwise = without_pointwise

        self.P = int(kernel_size * kernel_size - self.remove_center)
        self.K =  group * (kernel_size * kernel_size - self.remove_center)
        if dw_kernel_size is not None:
            self.offset_mask_dw = \
                nn.Conv2d(channels, channels, dw_kernel_size, stride=1, padding=(dw_kernel_size - 1) // 2, groups=channels)
        self.offset_mask = nn.Linear(channels, int(math.ceil((self.K * 3)/8)*8))
        # self.offset = nn.Linear(channels, self.K * 2)
        # self.mask = nn.Linear(channels, self.K)
        if not without_pointwise:
            self.value_proj = nn.Linear(channels, channels)
            self.output_proj = nn.Linear(channels, channels, bias=output_bias)
        self._reset_parameters()

    def _reset_parameters(self):
        constant_(self.offset_mask.weight.data, 0.)
        constant_(self.offset_mask.bias.data, 0.)
        if not self.without_pointwise:
            xavier_uniform_(self.value_proj.weight.data)
            constant_(self.value_proj.bias.data, 0.)
            xavier_uniform_(self.output_proj.weight.data)
            if self.output_proj.bias is not None:
                constant_(self.output_proj.bias.data, 0.)

    def forward(self, input):
        """
        :param input                       (N, L, C)
        :param shape                       (H, W) or None
        :return output                     (N, L, C)
        """
        N, C, H, W = input.shape
        L = H * W
        x = input.permute(0, 2, 3, 1).view(N, H * W, C)
        if not self.without_pointwise:
            x = self.value_proj(x)
        x = x.reshape(N, H, W, -1)
        if self.dw_kernel_size is not None:
            offset_mask_input = self.offset_mask_dw(input.reshape(N, H, W, C).permute(0, 3, 1, 2))
            offset_mask_input = offset_mask_input.permute(0, 2, 3, 1).view(N, L, C)
        else:
            offset_mask_input = input
        offset_mask = self.offset_mask(offset_mask_input).reshape(N, H, W, -1)
        offset_mask_no_pad = offset_mask[:, :, :, : self.K * 3]
        offset_mask_no_pad = offset_mask_no_pad.unflatten(-1, (self.group, self.P * 3))
        offset = offset_mask_no_pad[:, :, :, :, : self.P * 2].flatten(-2)
        mask = offset_mask_no_pad[:, :, :, :, self.P * 2: self.P * 3].flatten(-2)
        # offset = self.offset(offset_mask_input).reshape(N, H, W, -1)
        # mask = self.mask(offset_mask_input).reshape(N, H, W, -1)
        x = dcnv4_core_pytorch(
            x,
            offset,
            mask,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.stride,
            self.pad,
            self.pad,
            self.dilation,
            self.dilation,
            self.group,
            self.group_channels,
            self.offset_scale
        )

        # print(x.shape)

        # x = x.view(N, L, -1)
        if not self.without_pointwise:
            x = self.output_proj(x)
        x = x.permute(0, 3, 1, 2)
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class DCNv4Block(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.dcnv3 = DCNv4_pytorch(c2)
        self.cv3 = Conv(c2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv3(self.dcnv3(self.cv2(self.cv1(x))))
        # shortcut = x      # 128,40,40
        # x1 = self.cv1(x)  # torch.Size([2, 64, 40, 40])
        # x2 = self.cv2(x1) # torch.Size([2, 128, 40, 40])
        # x3 = self.dcnv3(x2) # torch.Size([2, 128, 40, 40])
        # x4 = self.cv3(x3) # torch.Size([2, 128, 40, 40])
        # return shortcut + x4


class DCNV4_Conv(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        # self.cv2 = Conv(c1, c2, 3, 1, g=g)
        self.cv2 = Conv(c1, c2, k=1)
        self.dcnv3 = DCNv4_pytorch(c2, group = 4)
        self.cv3 = Conv(c2, c2, 1, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        shortcut = x      # 128,40,40
        x2 = self.cv2(x) # torch.Size([2, 128, 40, 40])
        x3 = self.dcnv3(x2) # torch.Size([2, 128, 40, 40])
        return x3
        # x4 = self.cv3(x3) # torch.Size([2, 128, 40, 40])
        # return shortcut + x4


# class DCNv4Conv(nn.Module):
#     def __init__(self, inc, ouc, shortcut=True, g=1, e=0.5, act=True):  # ch_in, ch_out, shortcut, groups, expansion
#         super().__init__()
#         if inc != ouc:
#             self.stem_conv = Conv(inc, ouc, k=1)
#         self.dcnv4 = DCNv4_pytorch(ouc)
#         self.bn = nn.BatchNorm2d(ouc)
#         self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#     def forward(self, x):
#         if hasattr(self, 'stem_conv'):
#             x = self.stem_conv(x)
#         x = self.dcnv4(x)
#         x = self.act(self.bn(x))
#         return x

class CKDCNv4(nn.Module):
    """CKDCNv4 Block."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(DCNv4Block(self.c, self.c, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))
    
    
    
# class DCNV4_Conv(nn.Module):
#     def __init__(self, inc, ouc, k=1, s=1, p=None, g=1, d=1, act=True):
#         super().__init__()
        
#         if inc != ouc:
#             self.stem_conv = Conv(inc, ouc, k=1)
            
#         self.dcnv4 = DCNv4_pytorch(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=g, dilation=d)
#         # self.dcnv4 = DCNv4(ouc, kernel_size=k, stride=s, pad=autopad(k, p, d), group=g, dilation=d)
#         self.bn = nn.BatchNorm2d(ouc)
#         self.act = Conv.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    
#     def forward(self, x):
#         if hasattr(self, 'stem_conv'):
#             x = self.stem_conv(x) # B C H W
#         x = self.dcnv4(x)
#         x = self.act(self.bn(x))
#         return x