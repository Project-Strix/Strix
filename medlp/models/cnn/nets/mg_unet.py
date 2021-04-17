from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import numpy as np
from medlp.models.cnn import SEGMENTATION_ARCHI
from medlp.models.cnn.nets import DynUNet
from medlp.models.cnn.blocks.dynunet_block_ex import *


@SEGMENTATION_ARCHI.register('2D', 'mg_unet')
@SEGMENTATION_ARCHI.register('3D', 'mg_unet')
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
        res_block: bool = False,
        last_activation: Optional[str] = None,
        is_prunable: bool = False,
        dropout: Optional[float] = None,
        filters: Optional[Sequence[int]] = None,
    ):
        super(MG_Unet, self).__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            strides=strides,
            upsample_kernel_size=upsample_kernel_size,
            norm_name=norm_name,
            deep_supervision=False,
            deep_supr_num=1,
            res_block=res_block,
            last_activation=last_activation,
            is_prunable=is_prunable,
            filters=filters
        )
        self.conv_block = UnetResBlock if res_block else UnetBasicBlock
        self.num_groups = num_groups
        self.dropout = dropout
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
            1,  # mix feature at bottleneck
            self.norm_name,
            is_prunable=self.is_prunable,
        )
        self.downsamples = self.get_downsamples()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0, last_activation=last_activation)
        self.apply(self.initialize_weights)

    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(inp, out, kernel_size, strides, UnetUpBlock, upsample_kernel_size)

    def get_output_block(self, idx: int, last_activation: Optional[str] = None):
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
                    "dropout": self.dropout
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

        return out
