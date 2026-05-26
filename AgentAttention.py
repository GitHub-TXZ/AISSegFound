import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_

class AgentAttention3D(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 agent_num=64, window=4, **kwargs):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.agent_num = agent_num
        self.window = window

        self.dwc = nn.Conv3d(in_channels=dim, out_channels=dim, kernel_size=(3, 3, 3),
                             padding=1, groups=dim)

        self.pool = nn.AdaptiveAvgPool1d(output_size=agent_num)

        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 4, 4, 4))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window, 1, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window, 1))
        self.ad_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, 1, window))

        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 4, 4, 4))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window, 1, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window, 1, agent_num))
        self.da_bias = nn.Parameter(torch.zeros(1, num_heads, 1, 1, window, agent_num))

        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ad_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        trunc_normal_(self.da_bias, std=.02)

    def forward(self, x):
        b, n, c = x.shape
        d , h , w = 8, 16, 16
        # d = h = w = 4 #int(n ** (1/3))  # 特征图的深度、高度和宽度
        num_heads = self.num_heads
        head_dim = c // num_heads

        qkv = self.qkv(x).reshape(b, n, 3, c).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q_t = q.reshape(b,n,c).permute(0, 2, 1)
        agent_tokens = self.pool(q_t)

        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        # position_bias1 = nn.functional.interpolate(self.an_bias, size=(self.window, self.window, self.window), mode='trilinear')
        # position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # position_bias2 = (self.ah_bias + self.aw_bias + self.ad_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        # position_bias = position_bias1 + position_bias2

        position_bias = 0
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        # agent_bias1 = nn.functional.interpolate(self.na_bias, size=(self.window, self.window, self.window), mode='trilinear')
        # agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        # agent_bias2 = (self.ha_bias + self.wa_bias + self.da_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        # agent_bias = agent_bias1 + agent_bias2

        agent_bias = 0
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        # v_ = v.transpose(1, 2).reshape(b, d, h, w, c).permute(0, 4, 1, 2, 3)
        # x = x + self.dwc(v_).permute(0, 2, 3, 4, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

if __name__ == '__main__':
    X = torch.randn(1, 6144, 768)  # 假设输入为 (B, D*H*W, C)
    B, N, C = X.size()
    Model = AgentAttention3D(dim=C, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                             agent_num=64, window=4)
    out = Model(X)
    print(out.shape)
