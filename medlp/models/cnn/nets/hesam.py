from typing import List, Optional, Sequence, Union, Callable

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn import init
import numpy as np

from monai.networks.blocks import Convolution, UpSample
from medlp.models.cnn.nets.dynunet import DynUNet
from monai_ex.networks.layers import Act, Norm, Conv, Pool
from monai_ex.networks.blocks import ResidualUnitEx as ResidualUnit
from torch.nn.modules.activation import ReLU


class TwoConv(nn.Sequential):
    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
        groups: int = 1,
    ):
        super().__init__()

        conv_0 = Convolution(
            dim,
            in_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            padding=1,
            groups=groups,
        )
        conv_1 = Convolution(
            dim,
            out_chns,
            out_chns,
            act=act,
            norm=norm,
            dropout=dropout,
            padding=1,
            groups=groups,
        )
        self.add_module("conv_0", conv_0)
        self.add_module("conv_1", conv_1)


class Down(nn.Sequential):
    def __init__(
        self,
        dim: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
        groups: int = 1,
    ):
        super().__init__()

        max_pooling = Pool["MAX", dim](kernel_size=2)
        convs = TwoConv(dim, in_chns, out_chns, act, norm, dropout, groups=groups)
        self.add_module("max_pooling", max_pooling)
        self.add_module("convs", convs)


class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        dim: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        halves: bool = True,
        groups: int = 1,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels to be upsampled.
            cat_chns: number of channels from the decoder.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
            upsample: upsampling mode, available options are
                ``"deconv"``, ``"pixelshuffle"``, ``"nontrainable"``.
            halves: whether to halve the number of channels during upsampling.
        """
        super().__init__()
        self.groups = groups
        up_chns = in_chns // 2 if halves else in_chns
        self.upsample = UpSample(dim, in_chns, up_chns, 2, mode=upsample)
        self.convs = TwoConv(
            dim, cat_chns + up_chns, out_chns, act, norm, dropout, groups=groups
        )

    def group_cat(self, x, y):
        x_shape = list(x.shape)  # [batch, x*cx, *]
        x = x.reshape(
            [x_shape[0], self.groups, x_shape[1] // self.groups] + x_shape[2:]
        )  # (batch, n, c, h, w, d)
        y = y.reshape(
            [x_shape[0], self.groups, y.shape[1] // self.groups] + x_shape[2:]
        )
        x = torch.cat([x, y], dim=2)  # (batch, n, cx+cy, h, w, d)
        x = x.reshape([x_shape[0], -1] + x_shape[2:])  # (batch, n*(cx+cy), h, w, d)
        return x

    def forward(self, x: torch.Tensor, x_e: torch.Tensor):
        """

        Args:
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
        dimensions = len(x.shape) - 2
        sp = [0] * (dimensions * 2)
        for i in range(dimensions):
            if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                sp[i * 2 + 1] = 1
        x_0 = torch.nn.functional.pad(x_0, sp, "replicate")

        if self.groups > 1:
            x = self.convs(self.group_cat(x_e, x_0))
        else:
            x = self.convs(
                torch.cat([x_e, x_0], dim=1)
            )  # input channels: (cat_chns + up_chns)
        return x


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
                return torch.stack(
                    [
                        func2(func1(x_i, self.weights[i : i + 1].t()), self.bias[i])
                        for i, x_i in enumerate(torch.unbind(x1, dim=axis))
                    ],
                    dim=axis,
                )
            else:
                return torch.stack(
                    [
                        func1(x_i, self.weights[i : i + 1].t())
                        for i, x_i in enumerate(torch.unbind(x1, dim=axis))
                    ],
                    dim=axis,
                )

        return apply_along_axis(x, torch.mm, torch.add, 1)


class MultiChannelLinear2(nn.Module):
    def __init__(self, in_features: int, n_channels: int):
        super().__init__()
        self.in_features = in_features
        self.n_channels = n_channels

        self.weights = Parameter(torch.Tensor(self.n_channels, self.in_features))

        # initialize weights and biases
        nn.init.zeros_(self.weights)
        # nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init

    def forward(self, x):
        z = x * self.weights[None, :, :]  # B*C*feat
        z = torch.sum(z, dim=2)[:, :, None]  # B*C*1
        return z


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
        groups: int = 1,
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

        self.conv_0 = TwoConv(
            dimensions, in_channels, features[0], act, norm, dropout, groups=groups
        )
        self.down_1 = Down(
            dimensions, features[0], features[1], act, norm, dropout, groups=groups
        )
        self.down_2 = Down(
            dimensions, features[1], features[2], act, norm, dropout, groups=groups
        )
        self.down_3 = Down(
            dimensions, features[2], features[3], act, norm, dropout, groups=groups
        )

        self.upcat_3 = UpCat(
            dimensions,
            features[3],
            features[2],
            features[1],
            act,
            norm,
            dropout,
            upsample,
            halves=False,
            groups=groups,
        )
        self.upcat_2 = UpCat(
            dimensions,
            features[1],
            features[1],
            features[0],
            act,
            norm,
            dropout,
            upsample,
            halves=False,
            groups=groups,
        )
        self.upcat_1 = UpCat(
            dimensions,
            features[0],
            features[0],
            last_feature,
            act,
            norm,
            dropout,
            upsample,
            halves=False,
            groups=groups,
        )

        self.final_conv = Conv["conv", dimensions](
            last_feature, last_feature, kernel_size=1
        )
        self.gmp = globalmaxpool(((1,) * dimensions))
        self.residuals = nn.Sequential(
            ResidualUnit(
                dimensions=dimensions,
                in_channels=last_feature,
                out_channels=features[-1],
                adn_ordering=("NA", "N"),
                strides=2,
                act=Act.RELU,
                norm=Norm.BATCH,
            ),
            nn.ReLU(),
            ResidualUnit(
                dimensions=dimensions,
                in_channels=features[-1],
                out_channels=features[-1],
                adn_ordering=("NA", "N"),
                strides=1,
                act=Act.RELU,
                norm=Norm.BATCH,
            ),
            nn.ReLU(),
        )
        self.sam = nn.Sequential(
            globalavgpool((sam_size,) * dimensions),
            # nn.Conv2d(features[-1], features[-1], kernel_size=sam_size, groups=features[-1]),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,) * dimensions), features[-1]),
        )
        self.final_fc = nn.Linear(features[-1], out_channels)
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(module):
        name = module.__class__.__name__.lower()
        if "conv3d" in name or "conv2d" in name:
            nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif "norm" in name:
            nn.init.normal_(module.weight, 1.0, 0.02)
            nn.init.zeros_(module.bias)

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


# @CLASSIFICATION_ARCHI.register('2D', 'nnHESAM')
# @CLASSIFICATION_ARCHI.register('3D', 'nnHESAM')
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
            kernel_size=(3,) + (3,) * n_depth,
            strides=(1,) + (2,) * n_depth,
            upsample_kernel_size=(1,) + (2,) * n_depth,
            norm_name=norm,
            deep_supervision=False,
            deep_supr_num=1,
            res_block=False,
            output_bottleneck=True,
        )
        # self.latent_code = None
        # self.backbone.bottleneck.register_forward_hook(
        #     hook=self.get_latent()
        # )
        features = self.backbone.filters
        print(f"nnHESAM features: {features}.")

        self.gmp = globalmaxpool(((1,) * dimensions))
        self.residuals = nn.Sequential(
            ResidualUnit(
                dimensions=dimensions,
                in_channels=last_feature,
                out_channels=features[-1],
                strides=2,
                act="relu",
                norm="batch",
            ),
            ResidualUnit(
                dimensions=dimensions,
                in_channels=features[-1],
                out_channels=features[-1],
                strides=1,
                act="relu",
                norm="batch",
            ),
        )
        self.sam = nn.Sequential(
            globalavgpool((sam_size,) * dimensions),
            # nn.Conv2d(features[-1], features[-1], kernel_size=sam_size),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((sam_size,) * dimensions), features[-1]),
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
