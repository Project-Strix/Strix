from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import numpy as np
# from pymic.layer.activation import get_acti_func
# from pymic.layer.convolution import ConvolutionLayer, DepthSeperableConvolutionLayer
# from pymic.layer.deconvolution import DeconvolutionLayer, DepthSeperableDeconvolutionLayer
# from pymic.net.net2d.unet2dres import get_acti_func, get_deconv_layer, get_unet_block, PEBlock
from medlp.models.cnn.blocks.dynunet_block_ex import *
from medlp.models.cnn.nets import DynUNet

class MG_Unet(DynUNet):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_groups: int,
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        upsample_kernel_size: Sequence[Union[Sequence[int], int]],
        norm_name: str = "instance",
        deep_supervision: bool = False,
        deep_supr_num: int = 1,
        res_block: bool = False,
        last_activation: Optional[str] = None,
        is_prunable: bool = False,
    ):
        super(MG_Unet, self).__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            deep_supervision=deep_supervision,
            deep_supr_num=deep_supr_num,
            res_block=res_block,
            last_activation=last_activation,
            is_prunable=is_prunable
        )
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.num_groups = num_groups
        self.input_block = self.conv_block(
            self.spatial_dims,
            self.in_channels,
            self.filters[0],
            self.kernel_size[0],
            self.strides[0],
            self.num_groups,
            self.norm_name,
            is_prunable=self.is_prunable,
        )
        self.bottleneck = self.conv_block(
            self.spatial_dims,
            self.filters[-2],
            self.filters[-1],
            self.kernel_size[-1],
            self.strides[-1],
            1, #mix feature at bottleneck
            self.norm_name,
            is_prunable=self.is_prunable,
        )
        self.downsamples = self.get_downsamples()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0, last_activation=last_activation)
    
    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)
    
    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size)

    def get_output_block(self, idx: int, upsample: int = 0, last_activation: Optional[str] = None):
            if upsample > 1:
                return nn.Sequential(UnetOutBlock(self.spatial_dims, 
                                                  self.filters[idx], 
                                                  self.out_channels*self.num_groups,
                                                  self.num_groups,
                                                  is_prunable=self.is_prunable,), 
                                     nn.UpsamplingBilinear2d(scale_factor=upsample))
            else:
                return UnetOutBlock(
                    self.spatial_dims, 
                    self.filters[idx], 
                    self.out_channels*self.num_groups, 
                    self.num_groups,
                    activation=last_activation, 
                    is_prunable=self.is_prunable,
                )

    def get_module_list(
        self,
        in_channels: List[int],
        out_channels: List[int],
        kernel_size: Sequence[Union[Sequence[int], int]],
        strides: Sequence[Union[Sequence[int], int]],
        conv_block: nn.Module,
        upsample_kernel_size: Optional[Sequence[Union[Sequence[int], int]]] = None,
    ):
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "num_groups": self.num_groups,
                    "norm_name": self.norm_name,
                    "upsample_kernel_size": up_kernel,
                    "is_prunable": self.is_prunable,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "num_groups": self.num_groups,
                    "norm_name": self.norm_name,
                    "is_prunable": self.is_prunable,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    def forward(self, x):
        out = self.input_block(x)
        outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            outputs.append(out)
        out = self.bottleneck(out)
        upsample_outs = []
        for upsample, skip in zip(self.upsamples, reversed(outputs)):
            out = upsample(out, skip)
            upsample_outs.append(out)
        out = self.output_block(out)
        if self.training and self.deep_supervision:
            start_output_idx = len(upsample_outs) - 1 - self.deep_supr_num
            upsample_outs = upsample_outs[start_output_idx:-1][::-1]
            preds = [self.deep_supervision_heads[i](out) for i, out in enumerate(upsample_outs)]
            return [out] + preds
        return out

class MGNet(nn.Module):
    def __init__(self, params):
        super(MGNet, self).__init__()
        self.params = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.ft_groups = self.params['feature_grps']
        self.norm_type = self.params['norm_type']
        self.block_type= self.params['block_type']
        self.n_class   = self.params['class_num']
        self.acti_func = self.params['acti_func']
        self.dropout   = self.params['dropout']
        self.depth_sep_deconv= self.params['depth_sep_deconv']
        self.deep_spv  = self.params['deep_supervision']
        self.resolution_level = len(self.ft_chns)
        assert(self.resolution_level == 5 or self.resolution_level == 4)

        Block = get_unet_block(self.block_type)
        self.block1 = Block(self.in_chns, self.ft_chns[0], self.norm_type, self.ft_groups[0],
             self.acti_func, self.params)

        self.block2 = Block(self.ft_chns[0], self.ft_chns[1], self.norm_type, self.ft_groups[1],
             self.acti_func, self.params)

        self.block3 = Block(self.ft_chns[1], self.ft_chns[2], self.norm_type, self.ft_groups[2],
             self.acti_func, self.params)

        self.block4 = Block(self.ft_chns[2], self.ft_chns[3], self.norm_type, self.ft_groups[3],
             self.acti_func, self.params)

        if(self.resolution_level == 5):
            self.block5 = Block(self.ft_chns[3], self.ft_chns[4], self.norm_type, self.ft_groups[4],
                self.acti_func, self.params)

            self.block6 = Block(self.ft_chns[3] * 2, self.ft_chns[3], self.norm_type, self.ft_groups[3],
                self.acti_func, self.params)

        self.block7 = Block(self.ft_chns[2] * 2, self.ft_chns[2], self.norm_type, self.ft_groups[2],
             self.acti_func, self.params)

        self.block8 = Block(self.ft_chns[1] * 2, self.ft_chns[1], self.norm_type, self.ft_groups[1],
             self.acti_func, self.params)
        
        self.block9= Block(self.ft_chns[0] * 2, self.ft_chns[0], self.norm_type, self.ft_groups[0],
             self.acti_func, self.params)


        self.down1 = nn.MaxPool2d(kernel_size = 2)
        self.down2 = nn.MaxPool2d(kernel_size = 2)
        self.down3 = nn.MaxPool2d(kernel_size = 2)

        DeconvLayer = get_deconv_layer(self.depth_sep_deconv)
        if(self.resolution_level == 5):
            self.down4 = nn.MaxPool2d(kernel_size = 2)
            self.up1 = DeconvLayer(self.ft_chns[4], self.ft_chns[3], kernel_size = 2,
                dim = 2, stride = 2, groups = self.ft_groups[3], acti_func = get_acti_func(self.acti_func, self.params))
        self.up2 = DeconvLayer(self.ft_chns[3], self.ft_chns[2], kernel_size = 2,
                dim = 2, stride = 2, groups = self.ft_groups[2], acti_func = get_acti_func(self.acti_func, self.params))
        self.up3 = DeconvLayer(self.ft_chns[2], self.ft_chns[1], kernel_size = 2,
                dim = 2, stride = 2, groups = self.ft_groups[1], acti_func = get_acti_func(self.acti_func, self.params))
        self.up4 = DeconvLayer(self.ft_chns[1], self.ft_chns[0], kernel_size = 2,
                dim = 2, stride = 2, groups = self.ft_groups[0], acti_func = get_acti_func(self.acti_func, self.params))
        
        if(self.dropout):
            self.drop1 = nn.Dropout(p=0.1)
            self.drop2 = nn.Dropout(p=0.2)
            self.drop3 = nn.Dropout(p=0.3)
            self.drop4 = nn.Dropout(p=0.4)
            if(self.resolution_level == 5):
                self.drop5 = nn.Dropout(p=0.5)
           
        self.conv9= nn.Conv2d(self.ft_chns[0], self.n_class * self.ft_groups[0], 
            kernel_size = 3, padding = 1, groups = self.ft_groups[0])      

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape)==5):
            [N, C, D, H, W] = x_shape
            new_shape = [N*D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        f1 = self.block1(x)
        if(self.dropout):
             f1 = self.drop1(f1)
        d1 = self.down1(f1)

        f2 = self.block2(d1)
        if(self.dropout):
             f2 = self.drop2(f2)
        d2 = self.down2(f2)

        f3 = self.block3(d2)
        if(self.dropout):
             f3 = self.drop3(f3)
        d3 = self.down3(f3)

        f4 = self.block4(d3)
        if(self.dropout):
             f4 = self.drop4(f4)

        if(self.resolution_level == 5):
            d4 = self.down4(f4)
            f5 = self.block5(d4)
            if(self.dropout):
                f5 = self.drop5(f5)

            f5up  = self.up1(f5)
            f4cat = interleaved_concate(f4, f5up)
            f6    = self.block6(f4cat)
            f6up  = self.up2(f6)
            f3cat = interleaved_concate(f3, f6up)
        else:
            f4up  = self.up2(f4)
            f3cat = interleaved_concate(f3, f4up)
        f7    = self.block7(f3cat)
        f7up  = self.up3(f7)

        f2cat = interleaved_concate(f2, f7up)
        f8    = self.block8(f2cat)
        f8up  = self.up4(f8)
       
        f1cat = interleaved_concate(f1, f8up)
        f9    = self.block9(f1cat)
        
        output = self.conv9(f9)
        
        if(len(x_shape)==5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)

        output_list = torch.chunk(output, self.ft_groups[0], dim = 1)    
        return output_list
