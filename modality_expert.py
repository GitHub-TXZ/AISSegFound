import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair

def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()


class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


class Moality_Expert(nn.Module):
    """An implementation of FFN with Channel Aggregation.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        kernel_size (int): The depth-wise conv kernel size as the
            depth-wise convolution. Defaults to 3.
        act_type (str): The type of activation. Defaults to 'GELU'.
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(Moality_Expert, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.fc1 = nn.Conv3d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv3d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv3d(
            in_channels=feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)
        
        self.bdsa_conv = nn.Conv3d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)

        # self.decompose = nn.Conv3d(
        #     in_channels=self.feedforward_channels,  # C -> 1
        #     out_channels=1, kernel_size=1,
        # )
        # self.sigma = ElementScale(
        #     self.feedforward_channels, init_value=1e-5, requires_grad=True)
        # self.decompose_act = build_act_layer(act_type)

    # def feat_decompose(self, x):
    #     # x_d: [B, C, H, W] -> [B, 1, H, W]
    #     x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
    #     return x
    
    def bdsa(self, x):
        residual = x
        x  = self.bdsa_conv(torch.abs(x - torch.flip(x, [4])))
        # 沿着通道维度全局平均池化
        # x = F.adaptive_avg_pool3d(x, 1)
        x = torch.mean(x, dim=1, keepdim=True)
        # 应用sigmoid函数
        x = torch.sigmoid(x)
        x = x * residual
        x = x + residual
        return x
        

    def forward(self, x):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, 8, 16, 16).contiguous()
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        # x = self.feat_decompose(x)
        x = self.bdsa(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(B, C, N).permute(0, 2, 1).contiguous()
        return x
    
if __name__ == '__main__':
    # test ChannelAggregationFFN
    ffn = Moality_Expert(embed_dims=768, feedforward_channels=128)
    x = torch.randn(2, 2048, 768)
    out = ffn(x)
    print(out.shape)
    print('ChannelAggregationFFN test passed')

    # # 2D test
    # ffn = ChannelAggregationFFN(768, 256)
    # x = torch.randn(2, 768, 16, 16)
    # out = ffn(x)
    # print(out.shape)
    # print('ChannelAggregationFFN test passed')
    
    
    params = sum(p.numel() for p in ffn.parameters())    
    print(f'Number of parameters: {params / 1e6:.2f}M')
    
    