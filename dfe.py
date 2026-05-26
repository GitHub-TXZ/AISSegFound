import torch
import torch.nn as nn
"""
    论文地址：https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0303670
    论文题目：DAU-Net: Dual attention-aided U-Net for segmenting tumor in breast ultrasound images
    中文题目：DAU-Net：用于乳腺超声图像中肿瘤分割的双重注意力辅助 U-Net
    讲解视频：https://www.bilibili.com/video/BV1JYqqYiEum/
        1、CBAM 通过通道注意力（CAM）和空间注意力（SAM）捕获上下文感知特征和空间关系。
        2、PAM 通过卷积层和注意力机制丰富局部特征并捕获空间关系，两者结合增强了模型对局部特征的表示能力。
"""

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

class PAM(nn.Module):  # 定义位置注意力模块
    def __init__(self, in_channels):  # 初始化函数
        super(PAM, self).__init__()  # 调用父类的初始化方法
        self.query_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)  # 定义查询卷积
        self.key_conv = nn.Conv3d(in_channels, in_channels // 8, kernel_size=1)  # 定义键卷积
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)  # 定义值卷积
        self.gamma = nn.Parameter(torch.zeros(1))  # 定义可训练参数 gamma
        self.softmax = nn.Softmax(dim=-1)  # 定义 softmax 函数

    def forward(self, x):  # 前向传播函数
        batch_size, C, height, width, depth = x.size()  # 获取输入张量的尺寸

        proj_query = self.query_conv(x).view(batch_size, -1, height * width * depth).permute(0, 2, 1)  # 计算查询向量并调整形状 Q
        proj_key = self.key_conv(x).view(batch_size, -1, height * width * depth)  # 计算键向量 K
        energy = torch.bmm(proj_query, proj_key)  # 计算能量矩阵
        attention = self.softmax(energy)  # 对能量矩阵应用 softmax

        proj_value = self.value_conv(x).view(batch_size, -1, height * width * depth)  # 计算值向量 V
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # 计算加权值向量
        out = out.view(batch_size, C, height, width, depth)  # 调整输出形状

        out = self.gamma * out + x  # 将加权输出与输入相加

        return out  # 返回最终输出

class PCBAM(nn.Module):  # 定义 PCBAM 模块
    def __init__(self, in_channels, ratio=8):  # 初始化函数
        super(PCBAM, self).__init__()  # 调用父类的初始化方法
        self.channel_attention = ChannelAttentionModule(in_channels, ratio)  # 定义通道注意力模块
        self.spatial_attention = SpatialAttentionModule()  # 定义空间注意力模块
        self.position_attention = PAM(in_channels)  # 定义位置注意力模块

    def forward(self, x):  # 前向传播函数
        # CBAM
        x_c = self.channel_attention(x)  # 通过通道注意力模块
        x_s = self.spatial_attention(x_c)  # 通过空间注意力模块

        x_p = self.position_attention(x)  # 通过位置注意力模块

        out = x_s + x_p  # 将空间和位置注意力的结果相加
        return out  # 返回最终输出

class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=8):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, ratio)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        x_c = self.channel_attention(x)
        x_s = self.spatial_attention(x_c)
        out = x_s + x
        return out

class DFE(nn.Module):
    def __init__(self, Conv_in_channels, Trans_in_channels, ratio=8, N_Modalities=None, base_feature=None):
        super(DFE, self).__init__()
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
    model = DFE(Conv_in_channels=256, Trans_in_channels=768, ratio=8, N_Modalities=None).to(device)
    print("Input shape:", x1.shape, x2.shape)
    out_dfe = model(x1, x2)
    print("Output shape:", out_dfe.shape) # 4, 128, 8, 16, 16

    #计算参数量(M)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"总参数数量：{params / 1e6}M")

     # 计算参数量(M)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
   
    #计算参数量(M)和Folps（G)
    flops, params = profile(model, inputs=(x1,x2))
    print(f"总参数量：{params / 1e6}M")
    print(f"总Flops：{flops / 1e9}G")

     

    num_trials = 1
    total_time = 0

    for _ in range(num_trials):
        torch.cuda.synchronize()
        start_time = time.time()
        _ = model(x1, x2)
        torch.cuda.synchronize()
        total_time += time.time() - start_time

    avg_time = total_time / num_trials
    print(f"Average forward pass time over {num_trials} runs: {avg_time:.6f} seconds")
    
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")

        
    
    
    
    
    # 假设输入张量形状为 (batch_size, in_channels, height, width)
    # batch_size = 4
    # in_channels = 64
    # height = 32
    # width = 32
    # depth = 32
    #____________________________________________________________________________________________________________________

    # x = torch.randn(batch_size, in_channels, height, width, depth)
    # cbam = CBAM(in_channels=in_channels)

    # print("Input shape:", x.shape)
    # out_pcbam = cbam(x)
    # print("Output shape:", out_pcbam.shape)

    # from thop import profile
    # import numpy as np
    #____________________________________________________________________________________________________________________

    # # 随机生成一个输入张量
    # x = torch.randn(batch_size, in_channels, height, width, depth)
    # pcbam = PCBAM(in_channels=in_channels)

    # print("Input shape:", x.shape)
    # out_pcbam = pcbam(x)
    # print("Output shape:", out_pcbam.shape)
    # #计算下模型参数量(M)和Folps（G)
    # model_parameters = filter(lambda p: p.requires_grad, pcbam.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # print(f"总参数数量：{params / 1e6}M")
    # flops, params = profile(pcbam, inputs=(x,))
    # print(f"总参数量：{params / 1e6}M")
    # print(f"总Flops：{flops / 1e9}G")
    
    
   
    







