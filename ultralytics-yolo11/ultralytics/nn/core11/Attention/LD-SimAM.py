import torch
import torch.nn as nn

class LD_SimAM(nn.Module):
    def __init__(self, channels, directions=4):
        super(LD_SimAM, self).__init__()
        self.directions = directions
        
        # 方向卷积核组（0°,45°,90°,135°）
        self.direction_convs = nn.ModuleList([
            nn.Conv2d(1,1,kernel_size=3,padding=1,bias=False) 
            for _ in range(directions)
        ])
        
        # 初始化方向滤波器
        kernels = [
            [[0,0,0], [1,1,1], [0,0,0]],  # 水平
            [[1,0,0], [0,1,0], [0,0,1]],  # 45°
            [[0,1,0], [0,1,0], [0,1,0]],  # 垂直
            [[0,0,1], [0,1,0], [1,0,0]]   # 135°
        ]
        for i, conv in enumerate(self.direction_convs):
            conv.weight.data = torch.tensor(kernels[i], dtype=torch.float32).view(1,1,3,3)
            conv.weight.requires_grad = False  # 固定滤波器
        
        # 自适应融合权重
        self.fusion = nn.Conv2d(directions, 1, kernel_size=1)
        
    def forward(self, x):
        # 灰度化
        gray = x.mean(dim=1, keepdim=True)  # [B,1,H,W]
        
        # 多方向边缘响应
        dir_feats = []
        for conv in self.direction_convs:
            dir_feats.append(conv(gray))
        dir_feats = torch.cat(dir_feats, dim=1)  # [B,4,H,W]
        
        # 动态融合
        spatial_att = torch.sigmoid(self.fusion(dir_feats))
        
        # 与原注意力结合
        energy = ...  # 原SimAM计算
        return x * (energy * spatial_att)
    
    
if __name__ == '__main__':
    from torchinfo import summary
    model = LD_SimAM()
    print("Check output shape ...")
    summary(model, input_size=(1, 3, 224, 224))
    x = torch.rand(1, 3, 224, 224)
    y = model(x)
    print(y.shape)
    for i in y:
        print(i.shape)