import torch
import torch.nn as nn


class ChannelAttentionModule(nn.Module):  # 定义通道注意力模块
    def __init__(self, in_channels, ratio=8):  # 初始化函数，设置输入通道数和缩放比例
        super(ChannelAttentionModule, self).__init__()  # 调用父类的初始化方法
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # 定义自适应平均池化层
        self.max_pool = nn.AdaptiveMaxPool3d(1)  # 定义自适应最大池化层

        self.fc1 = nn.Conv3d(in_channels, in_channels // ratio, kernel_size=1, bias=False)  # 定义第一个全连接卷积层
        self.relu1 = nn.ReLU()  # 定义 ReLU 激活函数
        self.fc2 = nn.Conv3d(in_channels // ratio, in_channels, kernel_size=1, bias=False)  # 定义第二个全连接卷积层

    def forward(self, x):  # 前向传播函数
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 通过平均池化计算通道注意力
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 通过最大池化计算通道注意力
        out = avg_out + max_out  # 将平均池化和最大池化的结果相加
        return x * torch.sigmoid(out)  # 用 sigmoid 激活函数调整输入并返回

class SpatialAttentionModule(nn.Module):  # 定义空间注意力模块
    def __init__(self):  # 初始化函数
        super(SpatialAttentionModule, self).__init__()  # 调用父类的初始化方法
        self.conv1 = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)  # 定义卷积层

    def forward(self, x):  # 前向传播函数
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 计算输入的平均值
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 计算输入的最大值
        out = torch.cat([avg_out, max_out], dim=1)  # 将平均值和最大值拼接在一起
        out = self.conv1(out)  # 通过卷积层
        return x * torch.sigmoid(out)  # 用 sigmoid 激活函数调整输入并返回


class CVFI(nn.Module):
    def __init__(self, Conv_in_channels, Trans_in_channels, ratio=8, N_Modalities=None, base_feature=None):
        super(CVFI, self).__init__()
        self.N_Modalities = N_Modalities
        self.channel_attention = ChannelAttentionModule(Trans_in_channels, ratio)
        self.spatial_attention = SpatialAttentionModule()
        # self.position_attention = PAM(Conv_in_channels+Trans_in_channels)
        self.cat_fusion_linear = nn.Conv3d(Conv_in_channels+Trans_in_channels, Conv_in_channels+Trans_in_channels, kernel_size=1)
        self.out_proj = nn.Conv3d(2*(Conv_in_channels+Trans_in_channels),Conv_in_channels, kernel_size=1)


    def forward(self, x_c, x_t): 
        residual_x_c = x_c
    
        # 这里所涉及到的B维度仍然是BC相乘
        B2, N, C = x_t.size()   # 8, 16, 16
        x_t = x_t.permute(0, 2, 1) # B, C, N
        x_t = x_t.view(B2, C, 8, 16, 16).contiguous() # B, C, H, W, D
        
        residual_x_t = x_t
        
        B1, C1, H1, W1, D1 = x_c.size()  # 这里的B是真正的batch_size
        # up_h = H1 // 8
        # up_w = W1 // 16
        # up_d = D1 // 16
        # x_t = nn.functional.interpolate(x_t, size=(up_h, up_w, up_d), mode='trilinear', align_corners=True)
        # x_c = x_c.view(B1, self.N_Modalities, C1//self.N_Modalities, H1, W1, D1).contiguous().view(B1*self.N_Modalities, -1, H1, W1, D1)
        
        x_c = self.spatial_attention(x_c)
        x_t = self.channel_attention(x_t)
        
        x_c_t = torch.cat([residual_x_c, residual_x_t], dim=1)
        # x_c_t = self.position_attention(x_c_t)
        x_c_t = self.cat_fusion_linear(x_c_t)
        
        out = torch.cat([x_c, x_t, x_c_t], dim=1)
        
        out = self.out_proj(out)
        
        return out



if __name__ == '__main__':
    from thop import profile
    import numpy as np
    import time
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 随机生成一个输入张量
    x1 = torch.randn(4, 256, 8, 16, 16).to(device)
    x2 = torch.randn(4, 8*16*16, 768).to(device)
    model = CVFI(Conv_in_channels=256, Trans_in_channels=768, ratio=8, N_Modalities=None).to(device)
    print("Input shape:", x1.shape, x2.shape)
    out_dfe = model(x1, x2)
    print("Output shape:", out_dfe.shape) # 4, 128, 8, 16, 16
