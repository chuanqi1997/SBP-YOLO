import torch
import torch.nn as nn
import torch.nn.functional as F

class ONNXCompatibleCondConv2d(nn.Module):
    """ ONNX兼容的条件卷积实现 """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=0, dilation=1, groups=1, bias=False, num_experts=4):
        super().__init__()
        self.num_experts = num_experts
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        # 为每个专家创建独立的卷积层
        self.experts = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size, 
                      stride, padding, dilation, groups, bias)
            for _ in range(num_experts)
        ])
        
    def forward(self, x, routing_weights):
        """
        x: [B, C, H, W]
        routing_weights: [B, num_experts]
        """
        B = x.shape[0]
        
        # 计算每个专家的输出并进行加权
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # 生成权重mask [B, 1, 1, 1]
            weight_mask = routing_weights[:, i].view(B, 1, 1, 1)
            
            # 计算专家输出并加权
            expert_out = expert(x)
            weighted_out = expert_out * weight_mask
            expert_outputs.append(weighted_out)
        
        # 合并所有专家输出
        out = torch.stack(expert_outputs, dim=0).sum(dim=0)
        return out

class DynamicConv_Single(nn.Module):
    """ 动态卷积层(ONNX兼容版) """
    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding=0, dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = ONNXCompatibleCondConv2d(
            in_features, out_features, kernel_size, 
            stride, padding, dilation, groups, bias, num_experts)
        
    def forward(self, x):
        # 路由权重计算
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        return self.cond_conv(x, routing_weights)

class DynamicConv(nn.Module):
    """ 完整的动态卷积模块 """
    default_act = nn.SiLU()  # 默认激活函数
    
    def __init__(self, c1, c2, k=1, s=1, num_experts=4, p=None, g=1, d=1, act=True):
        super().__init__()
        padding = autopad(k, p, d)  # 确保autopad返回整数padding值
        
        self.conv = nn.Sequential(
            DynamicConv_Single(c1, c2, kernel_size=k, stride=s, 
                              padding=padding, dilation=d, groups=g, 
                              num_experts=num_experts),
            nn.BatchNorm2d(c2),
            self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        )
    
    def forward(self, x):
        return self.conv(x)
    
    def channel_shuffle(self, x, groups):
        # 保持原有实现，此操作ONNX兼容
        N, C, H, W = x.size()
        out = x.view(N, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)
        return out

# 辅助函数需要确保返回整数padding
def autopad(k, p=None, d=1):
    # 返回整数padding值，确保ONNX兼容
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p