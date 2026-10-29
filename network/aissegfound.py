import os
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
import math
from typing import Optional
from monai.networks.nets import ViT   # monai版本问题会报多线程断言错误？

try:
    from conv_moe_3d import MoEConv
    from modality_expert import Moality_Expert
    from gc_branch import GC_branch
    from cvfi import CVFI
    from cafmm import CAFMM
    from utils.modality_dropout import simulate_missing_modalities
    from tafmm import TAFMM
    from decoder import UNetDecoder
except:
    from .conv_moe_3d import MoEConv
    from .modality_expert import Moality_Expert
    from .gc_branch import GC_branch
    from .cvfi import CVFI
    from .cafmm import CAFMM
    from .utils.modality_dropout import simulate_missing_modalities
    from .tafmm import TAFMM
    from .decoder import UNetDecoder
    

class Conv_LoRA_qkv(nn.Module):
    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
            linear_a_k: Optional[nn.Module] = None,
            linear_b_k: Optional[nn.Module] = None,
            r = 4,
            patch_size = [4,16,16],
            img_size = [32,256,256],
            conv_lora=False
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        if linear_a_k is not None:
            self.linear_a_k = linear_a_k
            self.linear_b_k = linear_b_k
        else:
            self.linear_a_k = None
        self.dim = qkv.in_features
        self.conv_lora = conv_lora

        if self.conv_lora:
            self.conv_q = MoEConv(patch_size,img_size,d=r,scales = [1, 2, 4, 8], M = 4, K = 2)
            self.conv_v = MoEConv(patch_size,img_size,d=r,scales = [1, 2, 4, 8], M = 4, K = 2)
            if linear_a_k is not None:
                self.conv_k = MoEConv(patch_size,img_size,d=r,scales = [1, 2, 4, 8], M = 4, K = 2)


    def forward(self, x):
        qkv = self.qkv(x)  # B D H W 3*dim  qkv 

        if self.conv_lora:
            new_q = self.linear_b_q(self.conv_q(self.linear_a_q(x)))
            new_v = self.linear_b_v(self.conv_v(self.linear_a_v(x)))
        else:
            new_q = self.linear_b_q(self.linear_a_q(x))
            new_v = self.linear_b_v(self.linear_a_v(x))

        if self.linear_a_k is not None:
            if self.conv_lora:
                new_k = self.linear_b_k(self.conv_k(self.linear_a_k(x)))
            else:
                new_k = self.linear_b_k(self.linear_a_k(x))
                
            qkv[:, :, self.dim:-self.dim] += new_k

        qkv[:, :, : self.dim] += new_q
        qkv[:, :,  -self.dim:] += new_v 
        return qkv



class PMEM(nn.Module):
    def __init__(self, mlp, number_experts, embed_dims=768, feedforward_channels=128, kernel_size=3, act_type='GELU', ffn_drop=0.):
        '''
        Args:
            number_experts: 专家数量(等于模态数量)
            embed_dims: 输入维度
            feedforward_channels: 隐藏层维度
            kernel_size: 卷积核大小
            act_type: 激活函数类型
            ffn_drop: dropout概率
        '''
        super(PMEM, self).__init__()
        self.number_experts = number_experts
        self.mlp = mlp
        # 根据number_experts的数量，创建相应数量的Moality_Expert
        self.experts = nn.ModuleList([Moality_Expert(embed_dims, feedforward_channels, kernel_size, act_type, ffn_drop) for _ in range(number_experts)])
        # self.norm = nn.LayerNorm(embed_dims)
    def forward(self, x):
        bc, l, d = x.shape
        modality_share = self.mlp(x)
        # 首先将输入x拆分为每个模态的输入
        x_split = x.view(bc // self.number_experts, self.number_experts, l, d)
        # 对每个模态分别应用对应的专家
        modality_specific = torch.stack([expert(x_split[:, i]) for i, expert in enumerate(self.experts)], dim=1)
        # 将模态专家部分的输出重新reshape为原始形状
        modality_specific = modality_specific.view(bc, l, d)
        # 合并共享模态部分和模态专家部分
        x = modality_share + modality_specific
        return x



class AISSegFound(nn.Module):
    def __init__(self, r: int=4, conv_lora=True, img_size = [32,256,256], N_Modalities=3, dataset = 'Dataset001_Test', base_feature=8, random_missing=True, missing_modalities_list=None, **kwargs):
        super(AISSegFound, self).__init__()

        self.N_Modalities = N_Modalities  # 模态数量
        self.dataset = dataset # 根据数据集来做特定操作
        self.base_feature = base_feature  #决定了CNN branch的通道数，以及CAFMM的输入通道数
        self.patch_size = [4, 16, 16]
        self.img_size = img_size
        self.random_missing = random_missing
        self.missing_modalities_list = missing_modalities_list
        
        if self.dataset == 'Dataset001_Test' or self.dataset == '001':
            self.N_Modalities = 3
            self.input_conbine_CBF_CBV = nn.Conv3d(2, 1, kernel_size=3, bias=False, padding=1)
            self.input_conbine_Tmax_MTT = nn.Conv3d(2, 1, kernel_size=3, bias=False, padding=1)  

        # ViT branch
        vit= ViT(
            in_channels=1,
            img_size=(32,256,256),
            patch_size=(4,16,16),
            pos_embed="perceptron",
        )

        # pretrained weights
        vit_checkpoint = 'network/pretrained_weights/ViT_pretrain.ckpt'
        with open(vit_checkpoint, "rb") as f:
            state_dict = torch.load(f, map_location='cpu')['state_dict']
            encoder_dict = {k.replace('model.encoder.', ''): v for k, v in state_dict.items() if 'model.encoder.' in k}
        missing_keys, unexpected_keys = vit.load_state_dict(encoder_dict, strict=False)
        print(f'Missing keys: {missing_keys}')
        print(f'Unexpected keys: {unexpected_keys}')
        print(f'Image_encoder load param: {vit_checkpoint}') 
        
        # Frozen foundation model ViT encoder
        for n, value in vit.named_parameters():
            value.requires_grad = False   #首先冻结所有参数   decoder和组合微调都是在后面加上的，所以参数必定是需要梯度的
            if 'neck' in n:
                value.requires_grad = True

        # MoE finetuning
        self.w_As = []
        self.w_Bs = []
        for t_layer_i, blk in enumerate(vit.blocks):
            w_qkv_linear = blk.attn.qkv
            mlp = blk.mlp
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            # w_a_linear_k = nn.Linear(self.dim, r, bias=False)
            # w_b_linear_k = nn.Linear(r, self.dim, bias=False)
            
            # 不微调k
            w_a_linear_k = None
            w_b_linear_k = None
            
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            # self.w_As.append(w_a_linear_k)
            # self.w_Bs.append(w_b_linear_k)
            blk.attn.qkv = Conv_LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
                w_a_linear_k,
                w_b_linear_k,
                r,
                self.patch_size,
                self.img_size,    
                conv_lora=conv_lora
            )
            blk.mlp = PMEM(mlp, number_experts=self.N_Modalities)
        self.reset_parameters()

        self.vit_encoder = vit   # 编码器输出形状为(B*C, 2048, 768)

        # TAFMM
        self.tafmm = TAFMM(embedding_dim=768, volumn_size=[8, 16, 16], nhead=12, num_layers=2, method='TF')

        # CNN branch
        # base_feature代表第一个stage每个模态获得的通道数,例如，base_feature=8，N_modalities=3，那么第一个stage每个模态获得的通道数为8，最后一个stage为128*3=384
        self.cnn_branch = GC_branch(base_feature=self.base_feature, N_modalities=self.N_Modalities) 

        # CNN branch 五个stage的自适应特征融合模块
        self.cafmm_stages = nn.ModuleList([
            CAFMM(in_channels=base_feature, reduction_ratio=8, N_Modalities=self.N_Modalities),
            CAFMM(in_channels=base_feature * 2, reduction_ratio=8, N_Modalities=self.N_Modalities),
            CAFMM(in_channels=base_feature * 4, reduction_ratio=8, N_Modalities=self.N_Modalities),
            CAFMM(in_channels=base_feature * 8, reduction_ratio=8, N_Modalities=self.N_Modalities),
            CAFMM(in_channels=base_feature * 16, reduction_ratio=8, N_Modalities=self.N_Modalities)
        ]) 
       
        # CVFI 
        self.cvfi = CVFI(Conv_in_channels=16*self.base_feature, Trans_in_channels=768, ratio=8, N_Modalities=self.N_Modalities)
        
        # decoder
        self.mask_decoder = UNetDecoder(num_class=2, base_feature=self.base_feature, levels=4, norm_type='instance', bias=True)

    
    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)
            
    def forward(self, x: torch.Tensor):
        if self.dataset == 'Dataset001_Test' or self.dataset == '001':
            x = torch.cat((self.input_conbine_CBF_CBV(x[:,0:2]), self.input_conbine_Tmax_MTT(x[:,2:4]), x[:,4].unsqueeze(1)), dim=1) # 5模态变3模态    

        B, C, D, H, W = x.shape
        assert self.N_Modalities == C

        if self.training:
            missing_modalities_list = simulate_missing_modalities(self.N_Modalities, 0.5) # 0.5的概率模拟缺失模态
        else:
            if self.random_missing:
                missing_modalities_list = simulate_missing_modalities(self.N_Modalities, 0.5)  # 0.5的概率模拟缺失模态
            elif self.missing_modalities_list is not None:
                missing_modalities_list = self.missing_modalities_list  # 系统缺失
            else:
                missing_modalities_list = None  # 全模态训练
        
        # 输入端缺失模态模拟
        if missing_modalities_list:
            for i in missing_modalities_list:
                x[:, i, :, :, :] = 0
        
        cnn_stages_output = self.cnn_branch(x)
        
        cnn_stages_output_chunk = [list(torch.chunk(stage, self.N_Modalities, dim=1)) for stage in cnn_stages_output] # stage0 resolution最小，
        # GC特征端缺失模态模拟
        if missing_modalities_list:
            for i in missing_modalities_list:
                for j in range(5):
                    cnn_stages_output_chunk[j][i] = None   # 对于GC_branch的输出，缺失模态的特征置为None

        cnn_after_fusion = []
        for i, stage in enumerate(self.cafmm_stages):
            cnn_after_fusion.append(stage(*cnn_stages_output_chunk[i]))  # B,C,D,H,W

        
        # 输入端已经经过处理
        x = x.view(B*C, -1, D, H, W).contiguous()  # BC,1,D,H,W
        image_embeddings = self.vit_encoder(x)
        bottle_image_embedding  = image_embeddings[0]   # BC, 2048, 768    # 取出最后一层
        bottle_image_embedding = bottle_image_embedding.view(B, C, 8, 16, 16, 768).permute(1, 0, 5, 2, 3, 4)  # N_modalities, B, C, D, H, W
        
        # vit特征级缺失模态处理
        if missing_modalities_list:
            valid_embeddings = []
            for i in range(self.N_Modalities):
                if i not in missing_modalities_list:
                    valid_embeddings.append(bottle_image_embedding[i])
            bottle_image_embedding = torch.stack(valid_embeddings, dim=0)
  
        vit_after_fusion = self.tafmm(bottle_image_embedding) + torch.mean(bottle_image_embedding, dim=0) 

        new_bottleneck = self.cvfi(cnn_after_fusion[4], vit_after_fusion.view(B, 768, -1).permute(0,2,1))   # B, 128, 8, 16, 16
      
        output_logits = self.mask_decoder(new_bottleneck, cnn_after_fusion[0:4])

        return output_logits
  
         

if __name__ == "__main__":
  
    import time
    model = AISSegFound(r=4, conv_lora=True, img_size = [32,256,256], N_Modalities=5, dataset='Dataset001_Test', base_feature=8).cuda()
    x = torch.randn(1, 5, 32, 256, 256).cuda()  # nnUNet标准输入格式，(batch_size, channels, D, H, W) # 以ISLES2018数据集为例c=5
    
    # 打印每个层的参数信息
    for name, param in model.named_parameters():
        print(f"层名称: {name}, 参数形状: {param.shape}, 是否可训练: {param.requires_grad}")

    from thop import profile
    flops, params = profile(model, inputs=(x,))
    print(f"FLOPs: {flops / 1e9} GFLOPs")    # 987 GFLOPs
    print(f"参数量: {params / 1e6:.2f} M")    #
   
   
    # 输出model每层可学习参数，以及可学习参数总量
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # print(f"Layer: {name} | Size: {param.size()} | Number of parameters: {param.numel()}")
            total_params += param.numel()
    print(f"Total number of trainable parameters: {total_params / 1e6}M")    #12层ConvloRA为0.28M
  
    y_pred = model(x)
    print("输出的形状", y_pred.shape)
    
