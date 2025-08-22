import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv

class ReLU1(nn.Module):
    """自定义ReLU1模块"""
    def __init__(self):
        super(ReLU1, self).__init__()
        
    def forward(self, x):
        return torch.clamp(F.relu(x), 0.0, 1.0)
    
class SIAM(nn.Module):
    def __init__(self, channels = None,out_channels = None):
        super(SIAM, self).__init__()
        self.relu1 = ReLU1()  # 使用自定义模块
    
    def forward(self, x): # torch.Size([1, 256, 8, 8])
        b, c, h, w = x.size()
        # ========== Step 1: 生成双向查询 ==========
        z = x.mean(dim=1)  # (b, h, w)
        A_col = z.mean(dim=1, keepdim=True)  # (b, 1, w)
        A_row = z.mean(dim=2, keepdim=True)  # (b, h, 1)
        
        # ========== Step 2: 跨维度注意力计算 ==========
        # 列方向处理
        F_col = x.permute(0, 2, 1, 3)  # (b, h, c, w)
        m_col = torch.einsum('bhcw,bwz->bhc', F_col, A_col.permute(0, 2, 1)) / w
        
        # 行方向处理
        F_row = x.permute(0, 3, 1, 2)  # (b, w, c, h)
        m_row = torch.einsum('bwch,bhk->bwc', F_row, A_row) / h
        
        # ========== Step 3: 生成3D注意力图 ==========
        # 维度调整
        m_col = m_col.view(b, h, c, 1).permute(0, 2, 1, 3)  # (b, c, h, 1)
        m_row = m_row.view(b, w, c, 1).permute(0, 2, 3, 1)  # (b, c, 1, w)
                
        # 空间扩展
        Y_col = self.relu1(m_col.expand(-1, -1, -1, w))  # (b, c, h, w)
        Y_row = self.relu1(m_row.expand(-1, -1, h, -1))  # (b, c, h, w)
        
        # 合并结果
        Y = Y_col + Y_row
        return Y 


class SIAMBlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = SIAM(c,c)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2SIAM(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(*(SIAMBlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


if __name__ == "__main__":
    # Testing the SIAM module
    siam = SIAM()
    input_tensor = torch.randn(64, 256, 20, 20)  # Example input tensor with shape (batch size, channels, height, width) torch.Size([64, 256, 20, 20]) B,C,
    output_tensor = siam(input_tensor)
    print("Output shape:", output_tensor.shape)