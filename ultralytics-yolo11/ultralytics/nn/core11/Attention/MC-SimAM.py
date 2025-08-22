import torch
import torch.nn as nn

class MC_SimAM(nn.Module):
    def __init__(self, channels, reduction=16, e_lambda=1e-4):
        super(MC_SimAM, self).__init__()
        self.channels = channels
        self.e_lambda = e_lambda
        
        # 多尺度上下文分支
        self.ms_conv = nn.Sequential(
            nn.Conv2d(channels, channels//reduction, 3, padding=1, groups=channels//reduction),
            nn.ReLU(),
            nn.Conv2d(channels//reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 边缘增强分支
        self.edge_conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        nn.init.constant_(self.edge_conv.weight, 1.0/9)  # 均值滤波
        
        # 动态参数
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 原始SimAM能量计算
        x_mean = x.mean(dim=[2,3], keepdim=True)
        x_var = (x - x_mean).pow(2)
        energy = x_var / (4 * (x_var.sum(dim=[2,3], keepdim=True)/(h*w-1) + self.e_lambda) )+ 0.5
        
        # 多尺度上下文权重
        ms_weight = self.ms_conv(x)
        
        # 边缘特征增强
        edge_feat = self.edge_conv(x.mean(dim=1, keepdim=True))  # 灰度化后边缘提取
        edge_weight = torch.sigmoid(edge_feat - x_mean)
        
        # 动态融合
        combined_att = torch.sigmoid(energy * self.gamma + ms_weight + edge_weight * self.beta)
        
        return x * combined_att
    
    
if __name__ == '__main__':
    from torchinfo import summary
    model = MC_SimAM()
    print("Check output shape ...")
    summary(model, input_size=(1, 3, 224, 224))
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    for i in y:
        print(i.shape)