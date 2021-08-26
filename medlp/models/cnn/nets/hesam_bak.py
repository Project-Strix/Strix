from typing import Sequence, Union, Callable, Optional, Tuple

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np

from monai.networks.blocks import Convolution, UpSample
from monai.networks.layers import Act, Norm, Conv, Pool
from monai.networks.layers.convutils import same_padding
from monai.utils import ensure_tuple_rep


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

        conv_0 = Convolution(dim, in_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1, groups=groups)
        conv_1 = Convolution(dim, out_chns, out_chns, act=act, norm=norm, dropout=dropout, padding=1, groups=groups)
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
        self.convs = TwoConv(dim, cat_chns + up_chns, out_chns, act, norm, dropout, groups=groups)

    def group_cat(self, x, y):
        x_shape = list(x.shape)  # [batch, x*cx, *]
        x = x.reshape([x_shape[0], self.groups, x_shape[1] // self.groups] + x_shape[2:])  # (batch, n, c, h, w, d)
        y = y.reshape([x_shape[0], self.groups, y.shape[1] // self.groups] + x_shape[2:])
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
            x = self.convs(torch.cat([x_e, x_0], dim=1))  # input channels: (cat_chns + up_chns)
        return x


class ResidualUnit(nn.Module):
    """
    Residual module with multiple convolutions and a residual connection.

    For example:

    .. code-block:: python

        from monai.networks.blocks import ResidualUnit

        convs = ResidualUnit(
            dimensions=3,
            in_channels=1,
            out_channels=1,
            adn_ordering="AN",
            act=("prelu", {"init": 0.2}),
            norm=("layer", {"normalized_shape": (10, 10, 10)}),
        )
        print(convs)

    output::

        ResidualUnit(
          (conv): Sequential(
            (unit0): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
            (unit1): Convolution(
              (conv): Conv3d(1, 1, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
              (adn): ADN(
                (A): PReLU(num_parameters=1)
                (N): LayerNorm((10, 10, 10), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (residual): Identity()
        )

    Args:
        dimensions: number of spatial dimensions.
        in_channels: number of input channels.
        out_channels: number of output channels.
        strides: convolution stride. Defaults to 1.
        kernel_size: convolution kernel size. Defaults to 3.
        subunits: number of convolutions. Defaults to 2.
        adn_ordering: a string representing the ordering of activation, normalization, and dropout.
            Defaults to "NDA".
        act: activation type and arguments. Defaults to PReLU.
        norm: feature normalization type and arguments. Defaults to instance norm.
        dropout: dropout ratio. Defaults to no dropout.
        dropout_dim: determine the dimensions of dropout. Defaults to 1.

            - When dropout_dim = 1, randomly zeroes some of the elements for each channel.
            - When dropout_dim = 2, Randomly zero out entire channels (a channel is a 2D feature map).
            - When dropout_dim = 3, Randomly zero out entire channels (a channel is a 3D feature map).

            The value of dropout_dim should be no no larger than the value of `dimensions`.
        dilation: dilation rate. Defaults to 1.
        bias: whether to have a bias term. Defaults to True.
        last_conv_only: for the last subunit, whether to use the convolutional layer only.
            Defaults to False.
        padding: controls the amount of implicit zero-paddings on both sides for padding number of points
            for each dimension. Defaults to None.

    """

    def __init__(
        self,
        dimensions: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        subunits: int = 2,
        adn_ordering: Union[Sequence[str], str] = ["NDA", "NDA"],
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        bias: bool = True,
        last_conv_only: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
    ) -> None:
        super().__init__()
        self.dimensions = dimensions
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Sequential()
        self.residual = nn.Identity()
        self.adn_ordering = ensure_tuple_rep(adn_ordering, subunits)
        if not padding:
            padding = same_padding(kernel_size, dilation)
        schannels = in_channels
        sstrides = strides
        subunits = max(1, subunits)

        for su in range(subunits):
            conv_only = last_conv_only and su == (subunits - 1)
            unit = Convolution(
                dimensions,
                schannels,
                out_channels,
                strides=sstrides,
                kernel_size=kernel_size,
                adn_ordering=self.adn_ordering[su],
                act=act,
                norm=norm,
                dropout=dropout,
                dropout_dim=dropout_dim,
                dilation=dilation,
                bias=bias,
                conv_only=conv_only,
                padding=padding,
            )

            self.conv.add_module(f"unit{su:d}", unit)

            # after first loop set channels and strides to what they should be for subsequent units
            schannels = out_channels
            sstrides = 1

        # apply convolution to input to change number of output channels and size to match that coming from self.conv
        if np.prod(strides) != 1 or in_channels != out_channels:
            rkernel_size = kernel_size
            rpadding = padding

            if np.prod(strides) == 1:  # if only adapting number of channels a 1x1 kernel is used with no padding
                rkernel_size = 1
                rpadding = 0

            conv_type = Conv[Conv.CONV, dimensions]
            self.residual = conv_type(in_channels, out_channels, rkernel_size, strides, rpadding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res: torch.Tensor = self.residual(x)  # create the additive residual from x
        cx: torch.Tensor = self.conv(x)  # apply x to sequence of operations
        return cx + res  # add the residual to the output


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
        # sam_size: int = 6,
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
        avgpool: Callable = Pool[Pool.AVG, dimensions]

        self.conv_0 = TwoConv(dimensions, in_channels, features[0], act, norm, dropout, groups=groups)
        self.down_1 = Down(dimensions, features[0], features[1], act, norm, dropout, groups=groups)
        self.down_2 = Down(dimensions, features[1], features[2], act, norm, dropout, groups=groups)
        self.down_3 = Down(dimensions, features[2], features[3], act, norm, dropout, groups=groups)

        self.upcat_3 = UpCat(dimensions, features[3], features[2], features[1], act, norm, dropout, upsample, halves=False, groups=groups)
        self.upcat_2 = UpCat(dimensions, features[1], features[1], features[0], act, norm, dropout, upsample, halves=False, groups=groups)
        self.upcat_1 = UpCat(dimensions, features[0], features[0], last_feature, act, norm, dropout, upsample, halves=False, groups=groups)

        self.final_conv = Conv["conv", dimensions](last_feature, last_feature, kernel_size=1)
        self.gmp = globalmaxpool(((1,)*dimensions))
        self.residuals = nn.Sequential(
            ResidualUnit(
                dimensions=dimensions,
                in_channels=last_feature,
                out_channels=features[-1],
                adn_ordering=('NA', 'N'),
                strides=2,
                act=Act.RELU,
                norm=Norm.BATCH,
                ),
            nn.ReLU(),
            ResidualUnit(
                dimensions=dimensions,
                in_channels=features[-1],
                out_channels=features[-1],
                adn_ordering=('NA', 'N'),
                strides=1,
                act=Act.RELU,
                norm=Norm.BATCH,
            ),
            nn.ReLU(),
        )
        self.sam = nn.Sequential(
            avgpool(kernel_size=5, stride=2, padding=0),
            # nn.Conv2d(features[-1], features[-1], kernel_size=sam_size, groups=features[-1]),
            nn.Flatten(start_dim=2),
            MultiChannelLinear(np.prod((6,)*dimensions), features[-1])
        )
        self.final_fc = nn.Linear(features[-1], out_channels)
        # self.apply(self.initialize_weights)

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
