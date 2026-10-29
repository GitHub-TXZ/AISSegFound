import os
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'    
import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class TAFMM(nn.Module):
    def __init__(self, embedding_dim=768, volumn_size=[8,16,16], nhead=4, num_layers=2, method='TF'):
        super(TAFMM, self).__init__()
        self.embedding_dim = embedding_dim # 输入维度
        self.d_model = self.embedding_dim
        self.patch_dim =  [4, 8, 8]      # [2,4,4] # 3D自适应池化到这么大，会与volunme_size比较以确定下采样、上采样的倍数  #(D,H,W)
        self.method = method
        self.scale_factor = [v // p for v, p in zip(volumn_size, self.patch_dim)] #(D,H,W)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=nhead,
                                                   batch_first=True, dim_feedforward=self.d_model * 2)
        self.fusion_block = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(p=0.1)
        self.avgpool = nn.AdaptiveAvgPool3d((self.patch_dim[0], self.patch_dim[1], self.patch_dim[2]))
        self.upsample = DUpsampling3D(self.embedding_dim, self.scale_factor)
        # if method=='Token':
        #     self.fusion_token = nn.parameter.Parameter(torch.zeros((1, self.patch_dim ** 3, self.d_model)))

    def forward(self, all_content):

        n_modality = len(all_content)

        token_content = self.project(all_content)
        position_enc = PositionalEncoding(self.d_model, token_content.size(1))

        out = self.fusion_block(self.dropout(position_enc(token_content)))
        atten_map = self.reproject(out, n_modality, self.method)
        return self.atten(all_content, atten_map, n_modality)

    def project(self, all_content):
        n_modality = len(all_content)
        token_content_in = None
        for i in range(n_modality):
            content = self.avgpool(all_content[i])
            content = content.permute(0, 2, 3, 4, 1).contiguous()
            content2 = content.view(content.size(0), -1, self.embedding_dim)
            if i == 0:
                token_content_in = content2
            else:
                token_content_in = torch.cat([token_content_in, content2], dim=1)
        return token_content_in

    def reproject(self, atten_map, n_modality, method):
        n_patch = self.patch_dim[0] * self.patch_dim[1] * self.patch_dim[2]
        a_m0 = None
        for i in range(n_modality):
            atten_mapi = atten_map[:, n_patch*i : n_patch*(i+1), :].view(
                    atten_map.size(0),
                    self.patch_dim[0],
                    self.patch_dim[1],
                    self.patch_dim[2],
                    self.embedding_dim,
                )

            atten_mapi = atten_mapi.permute(0, 4, 1, 2, 3).contiguous()
            atten_mapi = self.upsample(atten_mapi).unsqueeze(dim=0)

            if a_m0 == None:
                a_m0 = atten_mapi
            else:
                a_m0 = torch.cat([a_m0, atten_mapi], dim=0)

        a_m = F.softmax(a_m0, dim=0)
        return a_m

    def atten(self, all_content, atten_map, n_modality):
        output = None
        for i in range(n_modality):
            a_m = atten_map[i, :, :, :, :, :]
            assert all_content[i].shape == a_m.shape, 'all_content and a_m cannot match!!'
            if output == None:
                output = all_content[i] * a_m
            else:
                output += all_content[i] * a_m
        return output


class DUpsampling3D(nn.Module):
    def __init__(self, inplanes, scale):
        super(DUpsampling3D, self).__init__()
        output_channel = inplanes * (scale[0] * scale[1] * scale[2])
        self.conv_3d = nn.Conv3d(inplanes, output_channel, kernel_size=1, stride=1, bias=False, groups=inplanes)
        self.scale = scale

    def forward(self, x):
        x = self.conv_3d(x)
        B, C, D, H, W = x.size()

        x_permuted = x.permute(0, 4, 3, 2, 1)

        x_permuted = x_permuted.contiguous().view((B, W, H, D * self.scale[0], int(C / (self.scale[0]))))

        x_permuted = x_permuted.permute(0, 3, 1, 2, 4)


        x_permuted = x_permuted.contiguous().view((B, D * self.scale[0], W, H * self.scale[1], int(C / (self.scale[0]*self.scale[1]))))

        x_permuted = x_permuted.permute(0, 1, 3, 2, 4)

        x_permuted = x_permuted.contiguous().view(
            (B, D * self.scale[0], H * self.scale[1], W * self.scale[2], int(C / (self.scale[0] * self.scale[1] * self.scale[2]))))

        x = x_permuted.permute(0, 4, 1, 2, 3)

        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach().to(torch.device('cuda'))



if __name__ == '__main__':
    import time
    from thop import profile
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = TAFMM(embedding_dim=768, volumn_size=[8, 16, 16], nhead=12, num_layers=2, method='TF').to(device)
    
    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params / 1e6:.2f}M")
   

    inputs = torch.randn(3, 1, 768, 8, 16, 16).to(device) #n-modality, batch_size, embedding_dim, volumn_size[0], volumn_size[1], volumn_size[2]
    y_pred = model(inputs)
    print(y_pred.shape)
        
        
