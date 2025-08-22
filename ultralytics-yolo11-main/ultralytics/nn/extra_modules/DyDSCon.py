import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.jit import Final
from torch.nn import init
from torch.nn.parameter import Parameter
import math
import numpy as np
from functools import partial
from typing import Optional, Callable, Union
from einops import rearrange, reduce
from ..modules.conv import Conv, DWConv, DSConv, RepConv, GhostConv, autopad
from ..modules.block import *
from .attention import *
from .rep_block import *
from .kernel_warehouse import KWConv
from .dynamic_snake_conv import DySnakeConv
from .ops_dcnv3.modules import DCNv3, DCNv3_DyHead
from .shiftwise_conv import ReparamLargeKernelConv
from .mamba_vss import *
from .fadc import AdaptiveDilatedConv
from .hcfnet import PPA, LocalGlobalAttention
from ..backbone.repvit import Conv2d_BN, RepVGGDW, SqueezeExcite
from ..backbone.rmt import RetBlock, RelPos2d
from ..backbone.inceptionnext import InceptionDWConv2d, MetaNeXtBlock
from .kan_convs import FastKANConv2DLayer, KANConv2DLayer, KALNConv2DLayer, KACNConv2DLayer, KAGNConv2DLayer
from .deconv import DEConv
from .SMPConv import SMPConv
from .camixer import CAMixer
from .orepa import *
from .RFAconv import *
from .wtconv2d import *
from .metaformer import *
from .tsdn import DTAB, LayerNorm

from ultralytics.utils.ops import make_divisible
from timm.layers import CondConv2d, trunc_normal_, use_fused_attn, to_2tuple
from timm.models import named_apply


# 原版动态卷积
class DynamicConv_Single(nn.Module):
    """ Dynamic Conv layer
    """
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                 groups, bias, num_experts)
        
    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
        return x


class DynamicDSConv_Single(nn.Module):
    """ Dynamic Depthwise Separable Conv layer """
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1,
                 num_experts=4, act_layer=nn.SiLU):
        super().__init__()
        # Depthwise部分参数
        self.dw_conv = DynamicConv_Single(
            in_features, 
            in_features, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_features,  # 关键修改：groups=in_features实现深度卷积
            num_experts=num_experts
        )
        
        # Pointwise部分参数
        self.pw_conv = nn.Sequential(
            nn.Conv2d(in_features, out_features, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_features)
        )
        
        # 路由网络共享
        self.routing = nn.Linear(in_features, num_experts)
        self.act_layer = act_layer
    def forward(self, x):
        # # 共享路由权重
        # pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        # routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        
        # Depthwise动态卷积
        x = self.dw_conv(x)
        
        # Pointwise固定卷积
        return self.act_layer(self.pw_conv(x))


class DynamicDSConv(nn.Module):
    """ 完整的动态深度可分离卷积模块 """
    default_act = nn.SiLU()  # 默认激活函数
    
    def __init__(self, c1, c2, k=1, s=1, num_experts=4, p=None, d=1, act=True):
        super().__init__()
        self.conv = DynamicDSConv_Single(
            c1, c2, 
            kernel_size=k, 
            stride=s, 
            padding=autopad(k, p, d), 
            dilation=d,
            num_experts=num_experts,
            act_layer=self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity
        )
    
    def forward(self, x):
        return self.conv(x)
    
    def channel_shuffle(self, x, groups):
        N, C, H, W = x.size()
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
        return out