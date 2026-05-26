import os
import torch.nn as nn
import torch
try:
    from Scconv import ScConv
except:
    from .Scconv import ScConv

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, N_modalities):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, groups=N_modalities)
        self.norm1 = nn.GroupNorm(num_groups=N_modalities, num_channels=out_channels) 
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, groups=N_modalities)
        # self.conv2 = ScConv(out_channels)
        self.norm2 = nn.GroupNorm(num_groups=N_modalities, num_channels=out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)
        residual = out
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)
        return out+residual

# MaxPool版
# def _build_branch(base_feature, N_modalities):
#     base_feature = base_feature
#     return nn.ModuleList([
#         ConvBlock3D(N_modalities, base_feature*N_modalities, N_modalities),
        
#         nn.Sequential(
#             ConvBlock3D(N_modalities, base_feature*N_modalities, N_modalities),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         ),
#         nn.Sequential(
#             ConvBlock3D(base_feature*N_modalities, base_feature*N_modalities*2, N_modalities),
#             nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
#         ),
#         nn.Sequential(
#             ConvBlock3D(base_feature*N_modalities*2, base_feature*N_modalities*4, N_modalities),
#             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
#         ),
#         nn.Sequential(
#             ConvBlock3D(base_feature*N_modalities*4, base_feature*N_modalities*8, N_modalities),
#             nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
#         )
#     ])
    
    
 # 卷积版   
def _build_branch(base_feature, N_modalities):
    base_feature = base_feature
    return nn.ModuleList([
        ConvBlock3D(N_modalities, base_feature*N_modalities, N_modalities),
        
        nn.Sequential(
            nn.Conv3d(base_feature*N_modalities, base_feature*N_modalities*2, kernel_size=3, stride=2, padding=1, groups=N_modalities),
            ConvBlock3D(base_feature*N_modalities*2, base_feature*N_modalities*2, N_modalities)
        ),
        nn.Sequential(
            nn.Conv3d(base_feature*N_modalities*2, base_feature*N_modalities*4, kernel_size=3, stride=2, padding=1, groups=N_modalities),
            ConvBlock3D(base_feature*N_modalities*4, base_feature*N_modalities*4, N_modalities)
        ),
        nn.Sequential(
            nn.Conv3d(base_feature*N_modalities*4, base_feature*N_modalities*8, kernel_size=3, stride=(1,2,2), padding=1, groups=N_modalities),
            ConvBlock3D(base_feature*N_modalities*8, base_feature*N_modalities*8, N_modalities)
        ),
        nn.Sequential(
            nn.Conv3d(base_feature*N_modalities*8, base_feature*N_modalities*16, kernel_size=3, stride=(1,2,2), padding=1, groups=N_modalities),
            ConvBlock3D(base_feature*N_modalities*16, base_feature*N_modalities*16, N_modalities)
        )
    ])

    

class GC_branch(nn.Module):
    def __init__(self, base_feature=None, N_modalities=None):
        super(GC_branch, self).__init__()
        self.branch = _build_branch(base_feature=base_feature, N_modalities=N_modalities)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.branch:
            if isinstance(layer, nn.Conv3d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)
                    
    def forward(self, x1):
        stage1 = self.branch[0](x1)
        stage2 = self.branch[1](stage1)
        stage3 = self.branch[2](stage2)
        stage4 = self.branch[3](stage3)
        stage5 = self.branch[4](stage4)
 
        return [stage1, stage2, stage3, stage4, stage5]



if __name__ == '__main__':
    
    from einops import rearrange
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "3"

    import numpy as np
    input_shape = (4, 3, 32, 256, 256)  # (batch_size, channels, height, width, depth)  # channels = N_modalities
    input_data = torch.randn(input_shape).cuda()
    N_modalities = 3
    gc_branch = GC_branch(base_feature=8, N_modalities=N_modalities).cuda()
    
    model_parameters = filter(lambda p: p.requires_grad, gc_branch.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"总参数数量：{params / 1e6}M")
    # 获取每个层的参数数量和名称
    for name, param in gc_branch.named_parameters():
        print(f"层名称: {name}, 参数形状: {param.shape}")

    total_params = sum(p.numel() for p in gc_branch.parameters())
    print(f"Total parameters: {total_params / 1e6}M")
    trainable_params = sum(p.numel() for p in gc_branch.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params / 1e6}M")
    output = gc_branch(input_data)
    print(output[4].shape)
    pass

    stages_output = gc_branch(input_data)
    stages_output = [torch.chunk(stage, N_modalities, dim=1) for stage in stages_output] # stage0 resolution最小，
    pass

