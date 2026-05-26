from mamba_ssm import Mamba
from monai.networks.layers.utils import get_act_layer, get_norm_layer
import torch.nn as nn
import torch
class MambaLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        # if x.dtype == torch.float16:
        #     x = x.type(torch.float32)
        # B, C = x.shape[:2]
        # assert C == self.input_dim
        # n_tokens = x.shape[2:].numel()
        # img_dims = x.shape[2:]
        # x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        # x_norm = self.norm(x_flat)
        x_norm = self.norm(x)
        # x_mamba = self.mamba(x_norm) + self.skip_scale * x_flat
        x_mamba = self.mamba(x_norm) + self.skip_scale * x
        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        # out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        # return out
        return x_mamba
    

if __name__ == "__main__":
    # 测试代码
    input_dim = 384
    output_dim = 384
    d_state = 16
    d_conv = 4
    expand = 2
    mamba_layer = MambaLayer(input_dim, output_dim, d_state, d_conv, expand)
    x = torch.randn(1, 8*16*16, 384)  # 输入数据
    output = mamba_layer(x)
    print(output.shape)  # 输出数据的形状