from typing import List, Optional, Sequence, Union, Callable

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from monai.networks.nets.basic_unet import TwoConv, Down, UpCat
from monai.networks.blocks import Convolution, UpSample
# from monai.networks.nets import DynUNet
from medlp.models.cnn import DynUNet
from monai_ex.networks.layers import Act, Norm, Conv, Pool
from monai_ex.networks.blocks import ResidualUnit
from torch.nn.modules.activation import ReLU
from medlp.models.cnn import CLASSIFICATION_ARCHI


class MultiChannelLinear(nn.Module):
    def __init__(self, in_features: int, n_channels: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.n_channels = n_channels

        self.weights = Parameter(torch.Tensor(self.n_channels, self.in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(self.n_channels))
        else:
            self.bias = None

        # initialize weights and biases
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        if bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        def apply_along_axis(x1, func1, func2, axis):
            if self.bias is not None:
                return torch.stack([
                    func2(func1(x_i, self.weights[i:i+1].t()), self.bias[i]) for i, x_i in enumerate(torch.unbind(x1, dim=axis))
                ], dim=axis)
            else:
                return torch.stack([
                    func1(x_i, self.weights[i:i+1].t()) for i, x_i in enumerate(torch.unbind(x1, dim=axis))
                ], dim=axis)
        return apply_along_axis(x, torch.mm, torch.add, 1)


@CLASSIFICATION_ARCHI.register('2D', 'HESAM')
@CLASSIFICATION_ARCHI.register('3D', 'HESAM')
class HESAM(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        features: Sequence[int] = (64, 128, 256, 256),
        last_feature: int = 64,
        sam_size: int = 6,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout=0.0,
        upsample: str = "deconv",
    ) -> None:
        """
        Args:
            dimensions: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            features: 4 integers as numbers of features.
                Defaults to ``(32, 64, 128, 256)``,
                - the first five values correspond to the five-level encoder feature sizes.
            last_feature: number of feature corresponds to the feature size after the last upsampling.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        """
        super().__init__()
        print(f"HESAM features: {features}.")
        globalmaxpool: Callable = Pool[Pool.ADAPTIVEMAX, dimensions]
        globalavgpool: Callable = Pool[Pool.ADAPTIVEAVG, dimensions]

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout)
        self.down_1 = Down(dimensions, features[0], features[1], act, norm, dropout)
        self.down_2 = Down(dimensions, features[1], features[2], act, norm, dropout)
        self.down_3 = Down(dimensions, features[2], features[3], act, norm, dropout)

        self.upcat_3 = UpCat(dimensions, features[3], features[2], features[1], act, norm, dropout, upsample, halves=False)
        self.upcat_2 = UpCat(dimensions, features[1], features[1], features[0], act, norm, dropout, upsample, halves=False)
        self.upcat_1 = UpCat(dimensions, features[0], features[0], last_feature, act, norm, dropout, upsample, halves=False)

        self.final_conv = Conv["conv", dimensions](last_feature, last_feature, kernel_size=1)
        self.gmp = globalmaxpool(((1,)*dimensions))
        self.residuals = nn.Sequential(
            ResidualUnit(
                dimensions=dimensions,
                in_channels=last_feature,
                out_channels=features[-1],
                strides=2,
                act=Act.RELU,
                norm=Norm.BATCH,
                ),
            ResidualUnit(
                dimensions=dimensions,
                in_channels=features[-1],
                out_channels=features[-1],
                strides=1,
                act=Act.RELU,
                norm=Norm.BATCH,
            )
        )
        self.sam = nn.Sequential(
            globalavgpool((sam_size,)*dimensions),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dimensions), features[-1])
        )
        self.final_fc = nn.Linear(features[-1], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.conv_0(x)

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)

        u3 = self.upcat_3(x3, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)

        out = self.final_conv(u1)
        out = self.residuals(out)
        out = self.sam(out)

        # high-level feature
        he = self.gmp(x3)
        hesam = torch.add(out, he.squeeze(dim=-1))
        logits = self.final_fc(hesam.squeeze())

        return logits


@CLASSIFICATION_ARCHI.register('2D', 'nnHESAM')
@CLASSIFICATION_ARCHI.register('3D', 'nnHESAM')
class HESAM2(nn.Module):
    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        last_feature: int = 64,
        sam_size: int = 6,
        act=Act.PRELU,
        norm="instance",
        dropout=0.0,
        upsample: str = "deconv",
    ) -> None:
        """
        Args:
            dimensions: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            last_feature: number of feature corresponds to the feature size after the last upsampling.
            act: activation type and arguments. Defaults to LeakyReLU.
            norm: feature normalization type and arguments. Defaults to instance norm.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
        """
        super().__init__()
        globalmaxpool: Callable = Pool[Pool.ADAPTIVEMAX, dimensions]
        globalavgpool: Callable = Pool[Pool.ADAPTIVEAVG, dimensions]
        n_depth = 3
        self.backbone = DynUNet(
            spatial_dims=dimensions,
            in_channels=in_channels,
            out_channels=last_feature,
            kernel_size=(3,)+(3,)*n_depth,
            strides=(1,)+(2,)*n_depth,
            upsample_kernel_size=(1,)+(2,)*n_depth,
            norm_name=norm,
            deep_supervision=False,
            deep_supr_num=1,
            res_block=False,
            output_bottleneck=True
        )
        # self.latent_code = None
        # self.backbone.bottleneck.register_forward_hook(
        #     hook=self.get_latent()
        # )
        features = self.backbone.filters
        print(f"nnHESAM features: {features}.")

        self.gmp = globalmaxpool(((1,)*dimensions))
        self.residuals = nn.Sequential(
            ResidualUnit(
                dimensions=dimensions,
                in_channels=last_feature,
                out_channels=features[-1],
                strides=2,
                act="relu",
                norm="batch"
                ),
            ResidualUnit(
                dimensions=dimensions,
                in_channels=features[-1],
                out_channels=features[-1],
                strides=1,
                act="relu",
                norm="batch"
            )
        )
        self.sam = nn.Sequential(
            globalavgpool((sam_size,)*dimensions),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,)*dimensions), features[-1])
        )
        self.final_fc = nn.Linear(features[-1], out_channels)

    def get_latent(self):
        def hook(model, input, output):
            self.latent_code = output
        return hook

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, latent_code = self.backbone(x)

        out = self.residuals(out)
        out = self.sam(out)

        # high-level feature
        he = self.gmp(latent_code)
        hesam = torch.add(out, he.squeeze(dim=-1))
        logits = self.final_fc(hesam.squeeze())

        return logits

