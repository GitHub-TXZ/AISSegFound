import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """通道注意力模块（Squeeze-and-Excitation Block）"""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

class MultiScaleFusion(nn.Module):
    """多尺度特征融合模块"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleFusion, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv3 = nn.Conv3d(in_channels, out_channels, kernel_size=7, padding=3)
        self.final_conv = nn.Conv3d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.final_conv(x)
        return x

class MultiModalDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        """
        Args:
            in_channels (int): 输入特征的通道数（384）。
            out_channels (int): 解码器中每层的通道数。
            num_classes (int): 输出类别数（2）。
        """
        super(MultiModalDecoder, self).__init__()
        self.num_classes = num_classes

        # 初始融合层
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(in_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.se_block = SEBlock(out_channels)  # 通道注意力

        # 多尺度特征融合
        self.multi_scale_fusion = MultiScaleFusion(out_channels, out_channels)

        # 解码器上采样阶段
        self.up1 = self._upsample_block(out_channels, out_channels // 2, l=1)  # 8x16x16 -> 16x32x32
        self.up2 = self._upsample_block(out_channels // 2, out_channels // 4, l=2)  # 16x32x32 -> 32x64x64
        self.up3 = self._upsample_block(out_channels // 4, out_channels // 8, l=3)  # 32x64x64 -> 64x128x128
        self.up4 = self._upsample_block(out_channels // 8, out_channels // 16, l=4)  # 64x128x128 -> 128x256x256

        # 最终输出层
        self.final_conv = nn.Conv3d(out_channels // 16, num_classes, kernel_size=1)

    def _upsample_block(self, in_channels, out_channels, l):
        """上采样模块"""
        if l==1 or l==2:
            return nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): 模态1的特征，形状为 [B, 384, 8, 16, 16]。
            x2 (torch.Tensor): 模态2的特征，形状为 [B, 384, 8, 16, 16]。
        Returns:
            torch.Tensor: 分割掩码，形状为 [B, 2, 32, 256, 256]。
        """
        # 特征融合
        x = torch.cat([x1, x2], dim=1)  # [B, 384*2, 8, 16, 16]
        x = self.fusion_conv(x)  # [B, out_channels, 8, 16, 16]
        x = self.se_block(x)  # 通道注意力
        x = self.multi_scale_fusion(x)  # 多尺度特征融合

        # 逐步上采样
        x = self.up1(x)  # [B, out_channels//2, 16, 32, 32]
        x = self.up2(x)  # [B, out_channels//4, 32, 64, 64]
        x = self.up3(x)  # [B, out_channels//8, 64, 128, 128]
        x = self.up4(x)  # [B, out_channels//16, 128, 256, 256]

        # 最终输出
        x = self.final_conv(x)  # [B, num_classes, 128, 256, 256]
        return x

# 测试代码
if __name__ == "__main__":
    # 输入特征
    B, C, D, H, W = 2, 384, 8, 16, 16
    x1 = torch.randn(B, C, D, H, W)  # 模态1的特征
    x2 = torch.randn(B, C, D, H, W)  # 模态2的特征

    # 初始化解码器
    decoder = MultiModalDecoder(in_channels=C, out_channels=256, num_classes=2)

    # 前向传播
    output = decoder(x1, x2)
    print("Output shape:", output.shape)  # 期望输出: [B, 2, 32, 256, 256]