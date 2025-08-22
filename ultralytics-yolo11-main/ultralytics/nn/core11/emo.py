# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import cv2
# import numpy as np
# from functools import partial

# from timm.models.efficientnet_builder import _parse_ksize
# from einops import rearrange, reduce
# from timm.models.layers.activations import *
# from timm.models.layers import DropPath, trunc_normal_, create_attn
# from timm.models.efficientnet_blocks import num_groups, SqueezeExcite as SE

# inplace = True

# class LayerNorm2d(nn.Module):

# 	def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
# 		super().__init__()
# 		self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

# 	def forward(self, x):
# 		x = rearrange(x, 'b c h w -> b h w c').contiguous()
# 		x = self.norm(x)
# 		x = rearrange(x, 'b h w c -> b c h w').contiguous()
# 		return x


# def get_norm(norm_layer='in_1d'):
# 	eps = 1e-6
# 	norm_dict = {
# 		'none': nn.Identity,
# 		'in_1d': partial(nn.InstanceNorm1d, eps=eps),
# 		'in_2d': partial(nn.InstanceNorm2d, eps=eps),
# 		'in_3d': partial(nn.InstanceNorm3d, eps=eps),
# 		'bn_1d': partial(nn.BatchNorm1d, eps=eps),
# 		'bn_2d': partial(nn.BatchNorm2d, eps=eps),
# 		# 'bn_2d': partial(nn.SyncBatchNorm, eps=eps),
# 		'bn_3d': partial(nn.BatchNorm3d, eps=eps),
# 		'gn': partial(nn.GroupNorm, eps=eps),
# 		'ln_1d': partial(nn.LayerNorm, eps=eps),
# 		'ln_2d': partial(LayerNorm2d, eps=eps),
# 	}
# 	return norm_dict[norm_layer]


# def get_act(act_layer='relu'):
# 	act_dict = {
# 		'none': nn.Identity,
# 		'sigmoid': Sigmoid,
# 		'swish': Swish,
# 		'mish': Mish,
# 		'hsigmoid': HardSigmoid,
# 		'hswish': HardSwish,
# 		'hmish': HardMish,
# 		'tanh': Tanh,
# 		'relu': nn.ReLU,
# 		'relu6': nn.ReLU6,
# 		'prelu': PReLU,
# 		'gelu': GELU,
# 		'silu': nn.SiLU
# 	}
# 	return act_dict[act_layer]


# class LayerScale(nn.Module):
# 	def __init__(self, dim, init_values=1e-5, inplace=True):
# 		super().__init__()
# 		self.inplace = inplace
# 		self.gamma = nn.Parameter(init_values * torch.ones(1, 1, dim))

# 	def forward(self, x):
# 		return x.mul_(self.gamma) if self.inplace else x * self.gamma


# class LayerScale2D(nn.Module):
# 	def __init__(self, dim, init_values=1e-5, inplace=True):
# 		super().__init__()
# 		self.inplace = inplace
# 		self.gamma = nn.Parameter(init_values * torch.ones(1, dim, 1, 1))

# 	def forward(self, x):
# 		return x.mul_(self.gamma) if self.inplace else x * self.gamma


# class ConvNormAct(nn.Module):

# 	def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
# 				 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
# 		super(ConvNormAct, self).__init__()
# 		self.has_skip = skip and dim_in == dim_out
# 		padding = math.ceil((kernel_size - stride) / 2)
# 		self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
# 		self.norm = get_norm(norm_layer)(dim_out)
# 		self.act = get_act(act_layer)(inplace=inplace)
# 		self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

# 	def forward(self, x):
# 		shortcut = x
# 		x = self.conv(x)
# 		x = self.norm(x)
# 		x = self.act(x)
# 		if self.has_skip:
# 			x = self.drop_path(x) + shortcut
# 		return x


# class iRMB(nn.Module):

# 	def __init__(self, dim_in, dim_out,  dim_head=64, attn_s=True, norm_in=True, has_skip=True, exp_ratio=1.0, norm_layer='bn_2d',
# 				 act_layer='relu', v_proj=True, dw_ks=3, stride=1, dilation=1, se_ratio=0.0,window_size=7,
# 				 qkv_bias=False, attn_drop=0., drop=0., drop_path=0., v_group=False, attn_pre=False):
# 		super().__init__()
# 		self.norm = get_norm(norm_layer)(dim_in) if norm_in else nn.Identity()
# 		dim_mid = int(dim_in * exp_ratio)
# 		self.has_skip = (dim_in == dim_out and stride == 1) and has_skip
# 		self.attn_s = attn_s
# 		if self.attn_s:
# 			assert dim_in % dim_head == 0, 'dim should be divisible by num_heads'
# 			self.dim_head = dim_head
# 			self.window_size = window_size
# 			self.num_head = dim_in // dim_head
# 			self.scale = self.dim_head ** -0.5
# 			self.attn_pre = attn_pre
# 			self.qk = ConvNormAct(dim_in, int(dim_in * 2), kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer='none')
# 			self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, groups=self.num_head if v_group else 1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
# 			self.attn_drop = nn.Dropout(attn_drop)
# 		else:
# 			if v_proj:
# 				self.v = ConvNormAct(dim_in, dim_mid, kernel_size=1, bias=qkv_bias, norm_layer='none', act_layer=act_layer, inplace=inplace)
# 			else:
# 				self.v = nn.Identity()
# 		self.conv_local = ConvNormAct(dim_mid, dim_mid, kernel_size=dw_ks, stride=stride, dilation=dilation, groups=dim_mid, norm_layer='bn_2d', act_layer='silu', inplace=inplace)
# 		self.se = SE(dim_mid, rd_ratio=se_ratio, act_layer=get_act(act_layer)) if se_ratio > 0.0 else nn.Identity()

# 		self.proj_drop = nn.Dropout(drop)
# 		self.proj = ConvNormAct(dim_mid, dim_out, kernel_size=1, norm_layer='none', act_layer='none', inplace=inplace)
# 		self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

# 	def forward(self, x):
# 		shortcut = x
# 		x = self.norm(x)
# 		B, C, H, W = x.shape
# 		if self.attn_s:
# 			# padding
# 			if self.window_size <= 0:
# 				window_size_W, window_size_H = W, H
# 			else:
# 				window_size_W, window_size_H = self.window_size, self.window_size
# 			pad_l, pad_t = 0, 0
# 			pad_r = (window_size_W - W % window_size_W) % window_size_W
# 			pad_b = (window_size_H - H % window_size_H) % window_size_H
# 			x = F.pad(x, (pad_l, pad_r, pad_t, pad_b, 0, 0,))
# 			n1, n2 = (H + pad_b) // window_size_H, (W + pad_r) // window_size_W
# 			x = rearrange(x, 'b c (h1 n1) (w1 n2) -> (b n1 n2) c h1 w1', n1=n1, n2=n2).contiguous()
# 			# attention
# 			b, c, h, w = x.shape
# 			qk = self.qk(x)
# 			qk = rearrange(qk, 'b (qk heads dim_head) h w -> qk b heads (h w) dim_head', qk=2, heads=self.num_head, dim_head=self.dim_head).contiguous()
# 			q, k = qk[0], qk[1]
# 			attn_spa = (q @ k.transpose(-2, -1)) * self.scale
# 			attn_spa = attn_spa.softmax(dim=-1)
# 			attn_spa = self.attn_drop(attn_spa)
# 			if self.attn_pre:
# 				x = rearrange(x, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
# 				x_spa = attn_spa @ x
# 				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
# 				x_spa = self.v(x_spa)
# 			else:
# 				v = self.v(x)
# 				v = rearrange(v, 'b (heads dim_head) h w -> b heads (h w) dim_head', heads=self.num_head).contiguous()
# 				x_spa = attn_spa @ v
# 				x_spa = rearrange(x_spa, 'b heads (h w) dim_head -> b (heads dim_head) h w', heads=self.num_head, h=h, w=w).contiguous()
# 			# unpadding
# 			x = rearrange(x_spa, '(b n1 n2) c h1 w1 -> b c (h1 n1) (w1 n2)', n1=n1, n2=n2).contiguous()
# 			if pad_r > 0 or pad_b > 0:
# 				x = x[:, :, :H, :W].contiguous()
# 		else:
# 			x = self.v(x)

# 		x = x + self.se(self.conv_local(x)) if self.has_skip else self.se(self.conv_local(x))

# 		x = self.proj_drop(x)
# 		x = self.proj(x)

# 		x = (shortcut + self.drop_path(x)) if self.has_skip else x
# 		return x
# # https://github.com/iscyy/ultralyticsPro
# from ultralytics.nn.modules.block import Conv, Bottleneck
# class CSRMBC(nn.Module):
#     # CSRMBC https://github.com/WongKinYiu/CrossStagePartialNetworks
#     def __init__(self, c1, c2, n=1, dim_head=64, attn_s=True, shortcut=True, k=(1, 1), g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super(CSRMBC, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c1, c_, k[0], 1)
#         self.cv3 = Conv(c_, c_, k[0], 1)
#         self.cv4 = Conv(2 * c_, c2, 1, 1)
#         self.m = nn.Sequential(*(iRMB(c_, c_, dim_head, attn_s) for _ in range(n)))

#     def forward(self, x):
#         y1 = self.cv3(self.m(self.cv1(x)))
#         y2 = self.cv2(x)
#         return self.cv4(torch.cat((y1, y2), dim=1))


# class RepConvN(nn.Module):
#     """RepConv is a basic rep-style block, including training and deploy status
#     This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
#     """
#     default_act = nn.SiLU()  # default activation

#     def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
#         super().__init__()
#         assert k == 3 and p == 1
#         self.g = g
#         self.c1 = c1
#         self.c2 = c2
#         self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

#         self.bn = None
#         self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
#         self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

#     def forward_fuse(self, x):
#         """Forward process"""
#         return self.act(self.conv(x))

#     def forward(self, x):
#         """Forward process"""
#         id_out = 0 if self.bn is None else self.bn(x)
#         return self.act(self.conv1(x) + self.conv2(x) + id_out)

#     def get_equivalent_kernel_bias(self):
#         kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
#         kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
#         kernelid, biasid = self._fuse_bn_tensor(self.bn)
#         return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

#     def _avg_to_3x3_tensor(self, avgp):
#         channels = self.c1
#         groups = self.g
#         kernel_size = avgp.kernel_size
#         input_dim = channels // groups
#         k = torch.zeros((channels, input_dim, kernel_size, kernel_size))
#         k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1.0 / kernel_size ** 2
#         return k

#     def _pad_1x1_to_3x3_tensor(self, kernel1x1):
#         if kernel1x1 is None:
#             return 0
#         else:
#             return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

#     def _fuse_bn_tensor(self, branch):
#         if branch is None:
#             return 0, 0
#         if isinstance(branch, Conv):
#             kernel = branch.conv.weight
#             running_mean = branch.bn.running_mean
#             running_var = branch.bn.running_var
#             gamma = branch.bn.weight
#             beta = branch.bn.bias
#             eps = branch.bn.eps
#         elif isinstance(branch, nn.BatchNorm2d):
#             if not hasattr(self, 'id_tensor'):
#                 input_dim = self.c1 // self.g
#                 kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
#                 for i in range(self.c1):
#                     kernel_value[i, i % input_dim, 1, 1] = 1
#                 self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
#             kernel = self.id_tensor
#             running_mean = branch.running_mean
#             running_var = branch.running_var
#             gamma = branch.weight
#             beta = branch.bias
#             eps = branch.eps
#         std = (running_var + eps).sqrt()
#         t = (gamma / std).reshape(-1, 1, 1, 1)
#         return kernel * t, beta - running_mean * gamma / std

#     def fuse_convs(self):
#         if hasattr(self, 'conv'):
#             return
#         kernel, bias = self.get_equivalent_kernel_bias()
#         self.conv = nn.Conv2d(in_channels=self.conv1.conv.in_channels,
#                               out_channels=self.conv1.conv.out_channels,
#                               kernel_size=self.conv1.conv.kernel_size,
#                               stride=self.conv1.conv.stride,
#                               padding=self.conv1.conv.padding,
#                               dilation=self.conv1.conv.dilation,
#                               groups=self.conv1.conv.groups,
#                               bias=True).requires_grad_(False)
#         self.conv.weight.data = kernel
#         self.conv.bias.data = bias
#         for para in self.parameters():
#             para.detach_()
#         self.__delattr__('conv1')
#         self.__delattr__('conv2')
#         if hasattr(self, 'nm'):
#             self.__delattr__('nm')
#         if hasattr(self, 'bn'):
#             self.__delattr__('bn')
#         if hasattr(self, 'id_tensor'):
#             self.__delattr__('id_tensor')

# class RepNBottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = RepConvN(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):  # ch_in, ch_out, shortcut, kernels, groups, expand
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, k[0], 1)
#         self.cv2 = Conv(c_, c2, k[1], 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

# class ReNRMB(nn.Module):
#     def __init__(self, c1, c2, n=1, dim_head=64, attn_s=True, isUse=True, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
#         if isUse:
#             self.m = nn.Sequential(*(RepNBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
#         else:
#             self.m = nn.Sequential(*(iRMB(c_, c_, dim_head, attn_s) for _ in range(n)))

#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

# class ReNLANRMB(nn.Module):
#     # ReNLANRMB Block
#     def __init__(self, c1, c2, c3, c4, c5, c=True, c6=1):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         self.c = c3//2
#         self.cv1 = Conv(c1, c3, 1, 1)
#         self.cv2 = nn.Sequential(ReNRMB(c3//2, c4, c6, dim_head=c5, attn_s=c, isUse=False))
#         self.cv3 = nn.Sequential(ReNRMB(c4, c4, c6, dim_head=c5, attn_s=c, isUse=False))
#         self.cv4 = Conv(c3+(2*c4), c2, 1, 1)

#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
#         return self.cv4(torch.cat(y, 1))

#     def forward_split(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
#         return self.cv4(torch.cat(y, 1))

# class CPNRMB(nn.Module):
#     def __init__(self, c1, c2, n=1, dim_head=64, attn_s=True, shortcut=True, g=1, e=0.5):
#         super().__init__()
#         self.c = int(c2 * e)  
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv(2 * self.c, c2, 1) 
#         self.m = nn.Sequential(*(iRMB(self.c, self.c, dim_head, attn_s) for _ in range(n)))
#     def forward(self, x):
#         a, b = self.cv1(x).chunk(2, 1)
#         return self.cv2(torch.cat((self.m(a), b), 1))

# class C3_RMB(nn.Module):
#     # C3_RMB Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, dim_head=64, attn_s=True, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
#         self.m = nn.Sequential(*(iRMB(c_, c_, dim_head, attn_s) for _ in range(n)))

#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

# class C2f_RMB(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1, c2, n=1, dim_head=64, attn_s=True, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  
#         self.m = nn.Sequential(*(iRMB(self.c, self.c, dim_head, attn_s) for _ in range(n)))

#     def forward(self, x):
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend(m(y[-1]) for m in self.m)
#         return self.cv2(torch.cat(y, 1))

