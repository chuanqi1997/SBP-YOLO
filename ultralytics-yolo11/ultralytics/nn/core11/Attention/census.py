import torch
import torch.nn as nn


# CENSUS Attention V1

class LearnableConvCensus(nn.Module):
    def __init__(self, in_channels, window_size=3):
        super().__init__()
        self.ws = window_size
        self.pad = (window_size-1)//2
        # 可学习参数：每个通道独立学习差值权重
        self.conv = nn.Conv2d(in_channels, in_channels*(window_size**2-1), 
                             window_size, padding=self.pad, groups=in_channels)
        # 可学习的温度参数（分通道）
        self.temperature = nn.Parameter(torch.ones(in_channels))

    def forward(self, x):
        B,C,H,W = x.shape
        # 计算差值 [B, C*(ws^2-1), H, W]
        diff = self.conv(x)
        # 重组为 [B,C,ws^2-1,H,W]
        diff = diff.view(B,C,-1,H,W)
        # 分通道应用温度参数
        binary = torch.sigmoid(diff * self.temperature.view(1,C,1,1,1))
        return binary.mean(dim=2)


class DiffConv(nn.Module):
    def __init__(self, window_size=3):
        super().__init__()
        self.ws = window_size
        self.pad = (window_size-1)//2
        # 定义非中心位置的卷积核
        self.conv = nn.Conv2d(1, window_size**2-1, window_size, 
                             padding=self.pad, groups=1, bias=False)
        self._init_weights()

    def _init_weights(self):
        """固定权重：邻域位置1，中心位置-1"""
        weight = torch.zeros((self.ws**2-1, 1, self.ws, self.ws))
        idx = 0
        center = (self.ws**2)//2
        for i in range(self.ws**2):
            if i != center:
                weight[idx, 0, i//self.ws, i%self.ws] = 1
                weight[idx, 0, center//self.ws, center%self.ws] = -1
                idx += 1
        self.conv.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x):
        # x: [B,C,H,W] → 分通道处理
        B,C,H,W = x.shape
        diff = self.conv(x.view(B*C,1,H,W))  # [B*C, ws^2-1, H, W]
        return diff.view(B,C,-1,H,W)         # [B,C,ws^2-1,H,W]

class ConvCensus(nn.Module):
    def __init__(self, window_size=3, temperature=1.0):
        super().__init__()
        self.diff_conv = DiffConv(window_size)
        self.temperature = temperature

    def forward(self, x):
        # 计算差值 [B,C,ws^2-1,H,W]
        diff = self.diff_conv(x) 
        # 软二值化
        binary = torch.sigmoid(diff * self.temperature)
        # 聚合 (均值)
        return binary.mean(dim=2)  # [B,C,H,W]


# #################################
class DifferentiableCensus(nn.Module):
    """可导的Census变换层（支持批量处理）"""
    def __init__(self, window_size=3, temperature=1.0):
        super().__init__()
        self.window_size = window_size
        self.pad = window_size // 2
        self.temperature = temperature  # 控制软二值化的平滑度

    def forward(self, x):
        B, C, H, W = x.shape
        # 扩展为多通道兼容（逐通道处理）
        x_padded = nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='replicate')
        # 提取局部窗口 [B, C, H, W, win_size, win_size]
        patches = x_padded.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
        patches = patches.contiguous().view(B, C, H, W, -1)  # [B, C, H, W, win*win]
        center = patches[..., (self.window_size**2) // 2].unsqueeze(-1)  # 中心像素值
        # 软二值化（可导）: 使用差异的符号 + Sigmoid平滑
        diff = patches - center
        binary = torch.sigmoid(diff * self.temperature)  # 近似二值化 [0~1]
        # 聚合为Census特征图 (均值池化)
        census_feat = binary.mean(dim=-1)  # [B, C, H, W]
        return census_feat

class CensusAttention(nn.Module):
    """Census Attention模块（可插入YOLO的Conv层中）"""
    def __init__(self, in_channels, reduction_ratio=16, window_size=3):
        super().__init__()
        
        # self.census = DifferentiableCensus(window_size=window_size)
        # self.census = LearnableConvCensus(in_channels = in_channels, window_size = window_size)
        self.census = ConvCensus(window_size = window_size) # 效果挺好

        # 注意力权重生成
        self.attn_net = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels // reduction_ratio, 8), 1),  # 压缩通道
            nn.ReLU(),
            nn.Conv2d(max(in_channels // reduction_ratio, 8), in_channels, 1),  # 恢复通道
            nn.Sigmoid()  # 输出0~1的注意力图
        )
        
    def forward(self, x): # torch.Size([128, 256, 20, 20]), B,C,X,Y 
        # 原始特征图
        conv_out = x  # 假设输入已经是卷积后的特征（若替换YOLO的Conv，需调整）
        # 计算Census注意力
        census_feat = self.census(x)  # [B, C, H, W]
        attn = self.attn_net(census_feat)  # [B, C, H, W]
        # 应用注意力
        return conv_out * attn + conv_out  # 残差连接增强稳定性


from ultralytics.nn.modules.conv import Conv


class CensusBlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = CensusAttention(c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x



class CensusPSA(nn.Module):

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(CensusBlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))



# class YOLOv11_CensusBlock(nn.Module):
#     """YOLOv11的改进Conv模块（替换原始Conv）"""
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
#         super().__init__()
#         # 原始卷积分支
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.SiLU(inplace=True)
#         )
#         # Census注意力分支
#         self.census_attn = CensusAttention(in_channels)  # 注意: 输入是卷积后的通道数
        
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.census_attn(x)  # 应用注意力
#         return x
    

# class DifferentiableCensus(nn.Module):
#     """可导的Census变换层（支持批量处理）"""
#     def __init__(self, window_size=3, temperature=1.0):
#         super().__init__()
#         self.window_size = window_size
#         self.pad = window_size // 2
#         self.temperature = temperature  # 控制软二值化的平滑度

#     def forward(self, x):
#         B, C, H, W = x.shape
#         # 优化填充方式为reflect
#         x_padded = nn.functional.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
#         # 提取局部窗口 [B, C, H, W, win_size, win_size]
#         patches = x_padded.unfold(2, self.window_size, 1).unfold(3, self.window_size, 1)
#         patches = patches.contiguous().view(B, C, H, W, -1)  # [B, C, H, W, win*win]
#         center = patches[..., (self.window_size**2) // 2].unsqueeze(-1)  # 中心像素值
#         # 软二值化（可导）: 使用差异的符号 + Sigmoid平滑
#         diff = patches - center
#         binary = torch.sigmoid(diff * self.temperature)  # 近似二值化 [0~1]
#         # 聚合为Census特征图 (均值池化)
#         census_feat = binary.mean(dim=-1)  # [B, C, H, W]
#         return census_feat

# class CensusAttention(nn.Module):
#     """Census Attention模块（可插入YOLO的Conv层中）"""
#     def __init__(self, in_channels, reduction_ratio=16, window_size=3):
#         super().__init__()
#         self.census = DifferentiableCensus(window_size=window_size)
        
#         # 注意力权重生成
#         self.attn_net = nn.Sequential(
#             nn.Conv2d(in_channels, max(in_channels // reduction_ratio, 8), 1),  # 压缩通道
#             nn.ReLU(),
#             nn.Conv2d(max(in_channels // reduction_ratio, 8), in_channels, 1),  # 恢复通道
#             nn.Sigmoid()  # 输出0~1的注意力图
#         )
        
#     def forward(self, x):
#         # 原始特征图
#         conv_out = x  # 假设输入已经是卷积后的特征（若替换YOLO的Conv，需调整）
        
#         # 计算Census注意力
#         census_feat = self.census(x)  # [B, C, H, W]
#         attn = self.attn_net(census_feat)  # [B, C, H, W]
        
#         # 应用注意力
#         return conv_out * attn + conv_out  # 残差连接增强稳定性   
    
    
    
    
    
    
    
    
    


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class EfficientCensus(nn.Module):
    """轻量级可导Census变换（通道降维 + 深度卷积优化）"""
    def __init__(self, in_channels=3, reduced_channels=8, window_size=3):
        super().__init__()
        self.window_size = window_size
        self.pad = window_size // 2
        
        # 通道降维 (输入RGB的3通道 -> 压缩到8通道)
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # 深度卷积替代unfold操作 (窗口大小为3x3)
        self.depth_conv = nn.Conv2d(
            in_channels=reduced_channels,
            out_channels=reduced_channels*(window_size**2-1),  # 每个通道生成 (win^2-1) 个特征
            kernel_size=window_size,
            padding=self.pad,
            groups=reduced_channels  # 深度可分离卷积
        )
        
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        """
        输入: [B, C, H, W] (C=3 for RGB)
        输出: [B, 1, H, W] (单通道Census特征图)
        """
        B, C, H, W = x.shape
        
        # 通道压缩
        x_reduced = self.channel_reducer(x)  # [B, reduced_C, H, W]
        
        # 深度卷积提取邻域特征 (等效于unfold)
        neighbors = self.depth_conv(x_reduced)  # [B, reduced_C*(win^2-1), H, W]
        
        # 重组为窗口形式
        neighbors = neighbors.view(
            B, self.channel_reducer[0].out_channels,
            (self.window_size**2 - 1), H, W
        )  # [B, reduced_C, win^2-1, H, W]
        
        # 中心像素值 (原始降维后的特征)
        center = x_reduced.unsqueeze(2)  # [B, reduced_C, 1, H, W]
        
        # 软二值化比较 (可导)
        binary = torch.sigmoid(
            (neighbors - center) * self.temperature.clamp(min=0.1)  # 防止温度趋零
        )  # [B, reduced_C, win^2-1, H, W]
        
        # 多通道聚合 (跨通道和窗口维度)
        census_feat = binary.mean(dim=[1,2])  # [B, H, W]
        return census_feat.unsqueeze(1)  # [B, 1, H, W]

class LightCensusAttention(nn.Module):
    """高效Census注意力模块（适用于YOLO的Conv替换）"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        # Census特征提取（优化版）
        self.census = EfficientCensus(in_channels=in_channels)
        
        # 注意力生成网络
        self.attn_net = nn.Sequential(
            nn.Conv2d(1, max(in_channels//reduction_ratio, 8), 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(max(in_channels//reduction_ratio, 8), 1, 1),
            nn.Sigmoid()
        )
        
        # 门控残差参数
        self.gate_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)  # 修改为in_channels
        
    def forward(self, x):
        """
        输入: [B, C, H, W] (来自YOLO的卷积特征)
        输出: [B, C, H, W]
        """
        # 生成Census注意力图
        census_map = self.census(x)  # [B, 1, H, W]
        attn = self.attn_net(census_map)  # [B, 1, H, W]
        
        # 门控残差机制
        gate = torch.sigmoid(self.gate_conv(x))  # [B, C, H, W]  # 修改为in_channels
        return x * (1 - gate) + (x * attn) * gate

class YOLOv11_EfficientBlock(nn.Module):
    """可直接替换YOLOv11原始Conv的模块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # 原始卷积分支
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )
        
        # 注意力分支（仅在通道数匹配时启用）
        if in_channels == out_channels:
            self.attention = LightCensusAttention(out_channels)
        else:  # 通道数变化时跳过注意力
            self.attention = nn.Identity()
        
    def forward(self, x):
        x = self.main_conv(x)
        return self.attention(x)
    
    
    
    
class MultiScaleDynamicCensus(nn.Module):
    """多尺度Census特征提取（含通道降维、深度卷积、动态温度）"""
    def __init__(self, in_channels=3, reduced_channels=8, window_sizes=[3,5]):
        super().__init__()
        self.window_sizes = window_sizes
        
        # 通道降维
        self.channel_reducer = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, 1),
            nn.BatchNorm2d(reduced_channels),
            nn.ReLU(inplace=True)
        )
        
        # 多尺度深度卷积组
        self.depth_convs = nn.ModuleList([
            nn.Conv2d(reduced_channels, reduced_channels*(ws**2-1), 
            kernel_size=ws, padding=ws//2, groups=reduced_channels)
            for ws in window_sizes
        ])
        
        # 动态温度参数（每个尺度独立）
        self.temperatures = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in window_sizes
        ])

    def forward(self, x):
        B, C, H, W = x.shape
        x_reduced = self.channel_reducer(x)  # [B, rc, H, W]
        
        multi_scale_feats = []
        for i, ws in enumerate(self.window_sizes):
            # 深度卷积提取邻域
            neighbors = self.depth_convs[i](x_reduced)  # [B, rc*(ws^2-1), H, W]
            neighbors = neighbors.view(B, -1, (ws**2-1), H, W)  # [B, rc, ws^2-1, H, W]
            
            # 中心像素值
            center = x_reduced.unsqueeze(2)  # [B, rc, 1, H, W]
            
            # 动态温度软二值化
            binary = torch.sigmoid((neighbors - center) * self.temperatures[i].clamp(min=0.1))
            scale_feat = binary.mean(dim=[1,2])  # [B, H, W]
            multi_scale_feats.append(scale_feat.unsqueeze(1))
        
        return torch.cat(multi_scale_feats, dim=1)  # [B, len(ws), H, W]

class ChannelSpatialDecoupledAttention(nn.Module):
    """空间-通道解耦注意力"""
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        # 通道注意力
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels//reduction_ratio, in_channels, 1),
            nn.Sigmoid())
        
        # 空间注意力融合层
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(len(self.window_sizes), 1, 3, padding=1),
            nn.Sigmoid())

    def forward(self, x, multi_scale_feats):
        # 通道注意力 [B, C, 1, 1]
        channel_attn = self.channel_attn(x)
        
        # 空间注意力 [B, 1, H, W]
        spatial_attn = self.spatial_fusion(multi_scale_feats)
        
        return channel_attn, spatial_attn

class EnhancedCensusAttention(nn.Module):
    """综合改进的Census注意力模块"""
    def __init__(self, in_channels, window_sizes=[3,5], reduction_ratio=16):
        super().__init__()
        # 多尺度Census特征提取
        self.multiscale_census = MultiScaleDynamicCensus(in_channels, window_sizes=window_sizes)
        
        # 空间-通道解耦注意力
        self.decoupled_attn = ChannelSpatialDecoupledAttention(in_channels, reduction_ratio)
        
        # 门控残差
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # 多尺度Census特征 [B, len(ws), H, W]
        census_feats = self.multiscale_census(x)
        
        # 解耦注意力
        channel_attn, spatial_attn = self.decoupled_attn(x, census_feats)
        
        # 注意力融合
        attn_output = x * channel_attn * spatial_attn
        
        # 门控残差
        gate = self.gate_conv(x)  # [B, 1, H, W]
        return x * (1 - gate) + attn_output * gate

class YOLOv11_EnhancedBlock(nn.Module):
    """可直接替换YOLOv11标准卷积的完整模块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # 主卷积分支
        self.main_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True))
        
        # 注意力分支（仅当通道数匹配时启用）
        if in_channels == out_channels:
            self.attention = EnhancedCensusAttention(out_channels)
        else:
            self.attention = nn.Identity()

    def forward(self, x):
        x = self.main_conv(x)
        return self.attention(x)

# if __name__ == "__main__":
#     # 测试模块
#     dummy_input = torch.randn(2, 3, 224, 224)
#     module = EnhancedCensusAttention(3)
#     output = module(dummy_input)
#     print(f"输入尺寸: {dummy_input.shape} => 输出尺寸: {output.shape}")
#     # 输出: 输入尺寸: torch.Size([2, 3, 224, 224]) => 输出尺寸: torch.Size([2, 3, 224, 224])
    
    