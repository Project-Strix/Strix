from typing import List, Optional, Sequence, Union, Callable

import torch
import torch.nn as nn
import numpy as np

from medlp.models.cnn.nets.dynunet import DynUNet
from medlp.models.cnn import MULTITASK_ARCHI
from monai.networks.layers.factories import Conv, Dropout, Norm, Pool
from monai.networks.layers.prunable_conv import PrunableLinear

@MULTITASK_ARCHI.registers('3D', 'unet_mlp')
class UNetMLP(DynUNet):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        seg_out_channels: int,
        cls_num_classes: int,
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
        super(UNetMLP, self).__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=seg_out_channels,
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

        if spatial_dims == 2:
            output_size = (7,7)
        elif spatial_dims == 3:
            output_size = (4,4,4) #(2,2,2) #For OOM issue
        else:
            raise ValueError(f'Only support 2D&3D data, but got dim = {spatial_dims}')

        pool_type: Callable = Pool[Pool.ADAPTIVEAVG, spatial_dims]
        fc_type: Callable = nn.Linear if not is_prunable else PrunableLinear
        self.avgpool = pool_type(output_size)
        num_ = np.prod(output_size) * self.filters[-1]
        self.classifier = nn.Sequential(
            fc_type(num_, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            fc_type(2048, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            fc_type(1024, cls_num_classes)
        )

    def forward(self, x):
        out = self.input_block(x)
        outputs = [out]
        for downsample in self.downsamples:
            out = downsample(out)
            outputs.append(out)
        out = self.bottleneck(out)

        cls_out = self.avgpool(out)
        cls_out = torch.flatten(cls_out, 1)
        cls_out = self.classifier(cls_out)

        upsample_outs = []
        for upsample, skip in zip(self.upsamples, reversed(outputs)):
            out = upsample(out, skip)
            upsample_outs.append(out)
        out = self.output_block(out)
        
        if self.training and self.deep_supervision:
            raise NotImplementedError
        
        return out, cls_out

