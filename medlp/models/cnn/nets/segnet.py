from typing import Sequence, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from monai_ex.networks.blocks import Convolution, UpSample
from monai_ex.networks.layers import Conv, Dropout, Norm, Pool
from medlp.models.cnn.nets.vgg import vgg16_bn
from medlp.models.cnn.utils import set_trainable


class MultiConv(nn.Sequential):
    """three convolutions."""
    def __init__(
        self,
        dim: int,
        num_conv: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        dropout: Union[float, tuple] = 0.0,
    ):
        """
        Args:
            dim: number of spatial dimensions.
            in_chns: number of input channels.
            out_chns: number of output channels.
            act: activation type and arguments.
            norm: feature normalization type and arguments.
            dropout: dropout ratio. Defaults to no dropout.
        """
        super().__init__()
        assert num_conv > 1, f'num_conv must larger than 1, but got {num_conv}'

        for idx in range(num_conv):
            kwargs = {"act": act, "norm": norm, "dropout": dropout, "padding": 1}
            if idx == 0:
                self.add_module(f"conv_{idx}", Convolution(dim, in_chns, out_chns, **kwargs))
            else:
                self.add_module(f"conv_{idx}", Convolution(dim, out_chns, out_chns, **kwargs))


class SegNet(nn.Module):
    def __init__(
        self,
        dim,
        in_channels,
        out_channels,
        pretrained=False,
        n_depth=5,
        freeze_bn=False,
        freeze_backbone=False,
        **kwargs,
    ):
        super().__init__()
        assert 1 < n_depth <= 5, f"Network depth should be (1, 5], but got {n_depth}"
        self.dim = dim
        self.output_latent = kwargs.get('output_bottleneck', False)
        self.use_fc = kwargs.get('use_fc', False)
        bn_out_channels = kwargs.get('bn_out_channels', 1)
        vgg_bn = vgg16_bn(pretrained=pretrained, dim=dim, num_classes=out_channels)
        ConvNd: Callable = Conv[Conv.CONV, dim]
        BatchNormNd: Callable = Norm[Norm.BATCH, dim]
        MaxPoolNd: Callable = Pool[Pool.MAX, dim]
        UnPoolNd: Callable = nn.MaxUnpool2d if dim == 2 else nn.MaxUnpool3d
        encoder = list(vgg_bn.features.children())
        encoder_stages = [slice(0, 6), slice(7, 13), slice(14, 23), slice(24, 33), slice(34, -1)]
        decoder_stages = [slice(0, 9), slice(9, 18), slice(18, 27), slice(27, 33), slice(33, -1)]

        encoder_stages = encoder_stages[:n_depth]
        decoder_stages = decoder_stages[5-n_depth:]

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = ConvNd(in_channels, 64, kernel_size=3, stride=1, padding=1)

        encoders = []
        for slice_ in encoder_stages:
            encoders.append(nn.Sequential(*encoder[slice_]))
        self.encoders = nn.ModuleList(encoders)
        self.pool = MaxPoolNd(kernel_size=2, stride=2, return_indices=True)

        self.bottleneck_feat_chns = 0
        for module in self.encoders:
            for layer in module:
                if isinstance(layer, ConvNd):
                    self.bottleneck_feat_chns = layer.out_channels

        # Decoder, same as the encoder but reversed, maxpool will not be used
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, MaxPoolNd)]
        # Replace the last conv layer
        decoder[-1] = ConvNd(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i+3][::-1]]
        # Replace some conv layers & batchN after them
        final_feat_chns = 0
        for i, module in enumerate(decoder):
            if isinstance(module, ConvNd):
                final_feat_chns = module.out_channels
                if module.in_channels != module.out_channels:
                    final_feat_chns = module.in_channels
                    decoder[i+1] = BatchNormNd(module.in_channels)
                    decoder[i] = ConvNd(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        decoders = []
        for slice_ in decoder_stages:
            decoders.append(nn.Sequential(*decoder[slice_]))
        self.decoders = nn.ModuleList(decoders)
        self.classifier = ConvNd(final_feat_chns, out_channels, kernel_size=3, stride=1, padding=1)
        self.unpool = UnPoolNd(kernel_size=2, stride=2)
        self.dense = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1,)*dim),
            ConvNd(self.bottleneck_feat_chns, bn_out_channels, kernel_size=1),
        )

        self._initialize_weights(*self.decoders)

        if freeze_bn:
            self.freeze_bn()

        if freeze_backbone:
            set_trainable(self.encoders, False)

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, (nn.Conv2d, nn.Conv3d)):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        sizes = []
        indices = []
        for encoder in self.encoders:
            x = encoder(x)
            # print('encode shape:', x.size())
            sizes.append(x.size())
            x, idx = self.pool(x)
            indices.append(idx)

        sizes = sizes[::-1]
        indices = indices[::-1]
        if self.output_latent:
            code = x.clone()

        # Decoder
        for i, decoder in enumerate(self.decoders):
            x = self.unpool(x, indices=indices[i], output_size=sizes[i])
            x = decoder(x)
            # print('decode shape:', x.size())

        x = self.classifier(x)

        if self.output_latent:
            if self.use_fc:
                y = self.dense(code).squeeze()
                return x, y
            return x, code
        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
