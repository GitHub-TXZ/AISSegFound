import torch
import torch.nn as nn
import torch.nn.functional as F
import einops


class LayerNormProxy3D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = einops.rearrange(x, 'b c d h w -> b d h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b d h w c -> b c d h w')


class DAttentionBaseline3D(nn.Module):
    def __init__(self, q_size, kv_size, n_heads, n_head_channels, n_groups,
                 attn_drop, proj_drop, stride,
                 offset_range_factor, use_pe, dwc_pe,
                 no_off, fixed_pe, ksize, log_cpb):
        super().__init__()
        self.dwc_pe = dwc_pe
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_d, self.q_h, self.q_w = q_size
        self.kv_d, self.kv_h, self.kv_w = self.q_d // stride, self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.use_pe = use_pe
        self.fixed_pe = fixed_pe
        self.no_off = no_off
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.log_cpb = log_cpb
        self.stride = stride
        kk = self.ksize
        pad_size = kk // 2 if kk != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv3d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size, groups=self.n_group_channels),
            LayerNormProxy3D(self.n_group_channels),
            nn.GELU(),
            nn.Conv3d(self.n_group_channels, 3, 1, 1, 0, bias=False)
        )
        if self.no_off:
            for m in self.conv_offset.parameters():
                m.requires_grad_(False)

        self.proj_q = nn.Conv3d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv3d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv3d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv3d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

    @torch.no_grad()
    def _get_ref_points(self, D_key, H_key, W_key, B, dtype, device):
        ref_z, ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, D_key - 0.5, D_key, dtype=dtype, device=device),
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_z, ref_y, ref_x), -1)
        ref[..., 2].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 1].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(D_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1, -1)  # B * g D H W 3

        return ref

    def forward(self, x):
        B, C, D, H, W = x.size()
        dtype, device = x.dtype, x.device

        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) d h w -> (b g) c d h w', g=self.n_groups, c=self.n_group_channels)
        offset = self.conv_offset(q_off).contiguous()  # B * g 3 Dg Hg Wg
        Dk, Hk, Wk = offset.size(2), offset.size(3), offset.size(4)
        n_sample = Dk * Hk * Wk

        if self.offset_range_factor >= 0 and not self.no_off:
            offset_range = torch.tensor([1.0 / (Dk - 1.0), 1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 3, 1, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)

        offset = einops.rearrange(offset, 'b p d h w -> b d h w p')
        reference = self._get_ref_points(Dk, Hk, Wk, B, dtype, device)

        if self.no_off:
            offset = offset.fill_(0.0)

        if self.offset_range_factor >= 0:
            pos = offset + reference
        else:
            pos = (offset + reference).clamp(-1., +1.)

        if self.no_off:
            x_sampled = F.avg_pool3d(x, kernel_size=self.stride, stride=self.stride)
            assert x_sampled.size(2) == Dk and x_sampled.size(3) == Hk and x_sampled.size(4) == Wk, f"Size is {x_sampled.size()}"
        else:
            x_sampled = F.grid_sample(
                input=x.reshape(B * self.n_groups, self.n_group_channels, D, H, W),
                grid=pos[..., (2, 1, 0)],  # z, y, x -> x, y, z
                mode='bilinear', align_corners=True)  # B * g, Cg, Dg, Hg, Wg

        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        q = q.reshape(B * self.n_heads, self.n_head_channels, D * H * W)
        k = self.proj_k(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)
        v = self.proj_v(x_sampled).reshape(B * self.n_heads, self.n_head_channels, n_sample)

        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, DHW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)

        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, D, H, W)

        y = self.proj_drop(self.proj_out(out))

        return y, pos.reshape(B, self.n_groups, Dk, Hk, Wk, 3), reference.reshape(B, self.n_groups, Dk, Hk, Wk, 3)


# 测试代码
q_size = (8, 16, 16)  # 输入特征图尺寸 (D, H, W)
kv_size = (8, 16, 16)  # 键值特征图尺寸（未使用）
n_heads = 8  # 注意力头数
n_head_channels = 16  # 每个头的通道数
n_groups = 4  # 分组数
attn_drop = 0.1  # 注意力 dropout
proj_drop = 0.1  # 输出投影 dropout
stride = 2  # 下采样步幅
offset_range_factor = 1.0  # 偏移范围因子
use_pe = True  # 使用位置编码
dwc_pe = False  # 不使用深度可分离卷积位置编码
no_off = False  # 使用偏移
fixed_pe = False  # 不使用固定位置编码
ksize = 3  # 卷积核大小
log_cpb = False  # 不使用对数坐标位置偏置

model = DAttentionBaseline3D(
    q_size, kv_size, n_heads, n_head_channels, n_groups,
    attn_drop, proj_drop, stride,
    offset_range_factor, use_pe, dwc_pe,
    no_off, fixed_pe, ksize, log_cpb
)

input_tensor = torch.randn(1, n_heads * n_head_channels, *q_size)  # (B, C, D, H, W)

# Forward pass
output, pos, reference = model(input_tensor)

# 打印输出信息
print("输出形状:", output.shape)
print("偏移形状:", pos.shape)
print("参考点形状:", reference.shape)
