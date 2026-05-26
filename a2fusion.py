import torch
from thop import profile
import torch.nn as nn
import torch.nn.functional as F

class AttentionFusionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, N_Modalities=None):
        """
        多输入注意力融合模块，支持任意数量的输入特征。
        
        参数:
        - in_channels: 输入特征的通道数
        - reduction_ratio: 注意力计算的通道压缩比例
        """
        super(AttentionFusionModule, self).__init__()
        self.in_channels = in_channels  #单个模态的嵌入的特征的数量
        self.N_Modalities = N_Modalities
        
        # 生成注意力映射的 1x1 卷积
        # 定义 N_Modalities 个 3D 卷积层
        self.modal_convs = nn.ModuleList([nn.Conv3d(in_channels*2, 1, kernel_size=3, padding=1) for _ in range(N_Modalities)])
        self.softmax = nn.Softmax(dim=1)
        # self.att_conv = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False)
        # self.att_fc = nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)

    def forward(self, *features):
        """
        前向传播，接受任意数量的输入特征。

        参数:
        - *features: 形状为 (B, C, H, W, D) 的多个输入张量

        返回:
        - 融合后的特征图 (B, C, H, W, D)
        """
        # B, C, H, W, D = features[0].shape
        num_features = len(features)
        assert num_features == self.N_Modalities, "输入特征数量与模态数量不匹配"
        # 找到features元组里不为None的所有元素，返回下标和值
        valid_features = [(i, features[i]) for i in range(num_features) if features[i] is not None]
        num_valid_features = len(valid_features)
        assert num_valid_features >= 1, "至少需要一个有效输入特征"
        
        # indices, features = zip(*valid_features) if valid_features else ([], [])
        avg_fusion = torch.mean(torch.stack([f for i, f in valid_features]), dim=0)
        # max_fusion = torch.max(torch.stack([f for i, f in valid_features]), dim=0)[0]
        att_maps = [self.modal_convs[i](torch.cat((avg_fusion, f), dim=1)) for i, f in valid_features]
        att_maps = torch.cat(att_maps, dim=1)  # (B, num_valid_features, H, W, D)
        att_maps = self.softmax(att_maps)  # 在第一个维度上进行 softmax

        fused = sum(att_maps[:, i].unsqueeze(1) * valid_features[i][1] for i in range(num_valid_features))
                
        return fused
        
        
        
        # att_maps[:, i].unsqueeze(1)
        
        # # 将输入拼接到一起 (B, num_features * C, H, W, D)
        # x = torch.cat(features, dim=1)

        # # 计算注意力映射
        # att = self.att_conv(x)  # (B, reduced_C, H, W, D)
        # att = F.relu(att)
        # att = self.att_fc(att)  # (B, num_features * C, H, W, D)
        # att = att.view(B, num_features, C, H, W, D)  # 重新调整形状
        # att = F.softmax(att, dim=1)  # 在 num_features 维度进行 softmax

        # # 加权融合
        # fused = sum(att[:, i] * features[i] for i in range(num_features))

        
if __name__ == "__main__":
    B, C, H, W, D = 2, 8, 32, 256, 256  # 设定输入尺寸
    feature1 = torch.randn(B, C, H, W, D)
    # feature2 = torch.randn(B, C, H, W, D) 
    feature2 = None 
    feature3 = torch.randn(B, C, H, W, D)  # 额外添加一个特征
    
    fusion_module = AttentionFusionModule(C, N_Modalities=3)
    output = fusion_module(feature1, None, feature3)
    # 计算参数量和FLOPs
    macs, params = profile(fusion_module, inputs=(feature1, None, feature3))

    print(f"参数量: {params / 1e6:.2f} M")
    print(f"FLOPs: {macs / 1e9:.2f} G")    
    print("输出形状:", output.shape)  # 应该仍然是 (B, C, H, W, D)



