from typing import Union, Tuple, Dict, Callable

import torch
import torch.nn as nn

from monai_ex.networks.layers import Conv, Norm, Pool
from monai.networks.blocks.squeeze_and_excitation import ChannelSELayer


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, dim=2):
    """3x3 convolution with padding"""
    conv_type: Callable = Conv[Conv.PRUNABLE_CONV, dim]
    return conv_type(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1, dim=2):
    """1x1 convolution"""
    conv_type: Callable = Conv[Conv.PRUNABLE_CONV, dim]
    return conv_type(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PrunableBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        dim=2,
    ):
        super(PrunableBasicBlock, self).__init__()
        self.dim = dim
        norm_type: Callable = Norm[Norm.BATCH, dim]
        conv_type: Callable = Conv[Conv.PRUNABLE_CONV, dim]
        maxpool_type: Callable = Pool[Pool.MAX, dim]
        avgpool_type: Callable = Pool[Pool.ADAPTIVEAVG, dim]

        if norm_layer is None:
            norm_layer = norm_type
        if groups != 1 or base_width != 64:
            raise ValueError(
                "PrunableBasicBlock only supports groups=1 and base_width=64"
            )
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in PrunableBasicBlock"
            )

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, dim=self.dim)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dim=self.dim)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBasicBlock(PrunableBasicBlock):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        dim=2,
        r: int = 2,
        acti_type_1: Union[Tuple[str, Dict], str] = ("relu", {"inplace": True}),
        acti_type_2: Union[Tuple[str, Dict], str] = "sigmoid",
    ):
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
            norm_layer=norm_layer,
            dim=dim,
        )

        self.se_layer = ChannelSELayer(
            spatial_dims=dim,
            in_channels=inplanes,
            r=r,
            acti_type_1=acti_type_1,
            acti_type_2=acti_type_2,
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        x = self.se_layer(x)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
