import torch
import torch.nn as nn


class UNetConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', norm_type='instance', bias=True, flag='encoder'):
        super(UNetConvBlock3D, self).__init__()
        if flag=='encoder':
            self.conv1 = ConvNormRelu3D(in_channels, out_channels//2, kernel_size=kernel_size, padding=padding,
                                      norm_type=norm_type, bias=bias)
            self.conv2 = ConvNormRelu3D(out_channels//2, out_channels, kernel_size=kernel_size, padding=padding,
                                      norm_type=norm_type, bias=bias)
        else:
            self.conv1 = ConvNormRelu3D(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                        norm_type=norm_type, bias=bias)
            self.conv2 = ConvNormRelu3D(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                        norm_type=norm_type, bias=bias)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs
    

class UNetConvBlock3D_hved(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding='SAME', norm_type='instance', bias=True, flag='encoder'):
        super(UNetConvBlock3D_hved, self).__init__()
        if flag=='encoder':
            self.conv1 = NormRelu3DConv(in_channels, out_channels//2, kernel_size=kernel_size, padding=padding,
                                      norm_type=norm_type, bias=bias)
            self.conv2 = NormRelu3DConv(out_channels//2, out_channels, kernel_size=kernel_size, padding=padding,
                                      norm_type=norm_type, bias=bias)
        else:
            self.conv1 = NormRelu3DConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                        norm_type=norm_type, bias=bias)
            self.conv2 = NormRelu3DConv(out_channels, out_channels, kernel_size=kernel_size, padding=padding,
                                        norm_type=norm_type, bias=bias)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class ConvNormRelu3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type='instance'):

        super(ConvNormRelu3D, self).__init__()
        norm = nn.BatchNorm3d if norm_type == 'batch' else nn.InstanceNorm3d
        if padding == 'same':
            p = padding
        elif padding == 'SAME':
            p = kernel_size // 2
        else:
            p = 0


        self.unit = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                            padding=p, stride=stride, bias=bias, dilation=dilation),
                                  norm(out_channels),
                                  nn.LeakyReLU(0.01, inplace=True))

    def forward(self, inputs):
        return self.unit(inputs)


class NormRelu3DConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding='SAME', bias=True, dilation=1, norm_type='instance'):

        super(NormRelu3DConv, self).__init__()
        norm = nn.BatchNorm3d if norm_type == 'batch' else nn.InstanceNorm3d
        if padding == 'SAME':
            p = kernel_size // 2
        else:
            p = 0

        self.unit = nn.Sequential(
            norm(in_channels),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=p, stride=stride, bias=bias, dilation=dilation)
        )

    def getpre_n(self, pre):
        re = {}
        p = pre.flatten()
        print(pre.shape)
        for i in range(len(p)):
            n = int(abs(p[i]))
            if n in re.keys():
                re[n] += 1
            else:
                re[n] = 1
        print(re)

    def forward(self, inputs):
        return self.unit(inputs)



class UNetDecoder(nn.Module):
    '''
    UNet decoder
    base feature maps: 8 与encoder的base feature maps相同
    '''
    def __init__(self, num_class, base_feature = 8, levels=4, norm_type='instance', bias=True):
        super(UNetDecoder, self).__init__()

        self.num_class = num_class
        self.feature_maps = base_feature
        self.levels = levels
        self.features = nn.Sequential()

        for i in range(levels):
            if i == 2 or i == 3:
                upconv = UNetUpSamplingBlock3D(2**(levels-i) * base_feature, 2**(levels-i-1) * base_feature, deconv=True,
                                            bias=bias, scale_factor=(2, 2, 2))
                self.features.add_module('upconv%d' % (i+1), upconv)
            elif i == 0 or i == 1:
                upconv = UNetUpSamplingBlock3D(2**(levels-i) * base_feature, 2**(levels-i-1) * base_feature, deconv=True,
                                            bias=bias, scale_factor=(1, 2, 2))
                self.features.add_module('upconv%d' % (i+1), upconv)

            conv_block = UNetConvBlock3D(2**(levels-i) * base_feature, 2**(levels-i-1) * base_feature,
                                           norm_type=norm_type, bias=bias, flag='decoder')

            self.features.add_module('convblock%d' % (i+1), conv_block)

        self.seg_head = nn.Conv3d(base_feature, num_class, kernel_size=1, stride=1, bias=bias)

    def forward(self, inputs, encoder_outputs):
        encoder_outputs.reverse()
        outputs = inputs
        for i in range(self.levels):
            outputs = getattr(self.features, 'upconv%d' % (i+1))(outputs)
            outputs = torch.cat([encoder_outputs[i], outputs], dim=1)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
        encoder_outputs.reverse()
        return self.seg_head(outputs)

class UNetDecoder_hved(nn.Module):
    def __init__(self, out_channels, feature_maps=64, levels=4, norm_type='instance', bias=True):
        super(UNetDecoder_hved, self).__init__()
        self.out_channels = out_channels
        self.feature_maps = feature_maps
        self.levels = levels
        self.features = nn.Sequential()

        for i in range(levels-1):
            upconv = UNetUpSamplingBlock3D(2**(levels-i-1) * feature_maps, 2**(levels-i-1) * feature_maps, deconv=False,
                                         bias=bias)
            self.features.add_module('upconv%d' % (i+1), upconv)

            conv_block = UNetConvBlock3D_hved(2**(levels-i-2) * feature_maps * 3, 2**(levels-i-2) * feature_maps,
                                           norm_type=norm_type, bias=bias, flag='decoder')

            self.features.add_module('convblock%d' % (i+1), conv_block)

        self.score = nn.Conv3d(feature_maps, out_channels, kernel_size=1, stride=1, bias=bias)

    def forward(self, inputs, encoder_outputs):
        encoder_outputs.reverse()
        outputs = inputs
        for i in range(self.levels-1):
            outputs = getattr(self.features, 'upconv%d' % (i+1))(outputs)
            outputs = torch.cat([encoder_outputs[i], outputs], dim=1)
            outputs = getattr(self.features, 'convblock%d' % (i+1))(outputs)
        encoder_outputs.reverse()
        return self.score(outputs)
    


class UNetUpSamplingBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, deconv=False, bias=True, scale_factor=None):
        super(UNetUpSamplingBlock3D, self).__init__()
        self.deconv = deconv
        # scale_factor元组中的每个元素减一
        # 还是元组
        padding = tuple(x - 1 for x in scale_factor)
        if self.deconv:
            self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor, padding=0)
            # self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        else:
            self.up = nn.Upsample(scale_factor=scale_factor, mode='trilinear', align_corners=True)

    def forward(self, *inputs):
        if len(inputs) == 2:
            return self.forward_concat(inputs[0], inputs[1])
        else:
            return self.forward_standard(inputs[0])

    def forward_concat(self, inputs1, inputs2):
        return torch.cat([inputs1, self.up(inputs2)], 1)

    def forward_standard(self, inputs):
        return self.up(inputs)


if __name__ == '__main__':
    from thop import profile
    base_feature=16
    input = torch.randn(4, 16*base_feature, 8, 16, 16) # bottleneck
    encoder_outputs = [torch.randn(4, base_feature, 32, 256, 256), torch.randn(4, 2*base_feature, 16, 128, 128), torch.randn(4, 4*base_feature, 8, 64, 64), torch.randn(4, 8*base_feature, 8, 32, 32)] # encoder skip-connects

    decoder = UNetDecoder(num_class=2, base_feature=base_feature, levels=4, norm_type='instance', bias=True)
    output = decoder(input, encoder_outputs)
   
    # 计算参数量和FLOPs
    macs, params = profile(decoder, inputs=(input,  encoder_outputs))

    print(f"参数量: {params / 1e6:.2f} M")
    print(f"FLOPs: {macs / 1e9:.2f} G")    
    print("输出形状:", output.shape)  # 应该仍然是 (B, C, H, W, D)
    print(f"Memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
