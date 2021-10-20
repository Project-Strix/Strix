import torch
import torch.nn as nn
from monai.networks.blocks.dynunet_block import get_conv_layer


class AnatomicalAttentionGate(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        featmap1_inchn: int,
        featmap2_inchn: int
    ):
        super().__init__()
        self.conv1 = get_conv_layer(
            spatial_dims,
            featmap1_inchn+featmap2_inchn,
            featmap1_inchn,
            kernel_size=1,
            stride=1,
            act='sigmoid',
            norm=None,
            bias=True,
            conv_only=False
        )

        self.conv2 = get_conv_layer(
            spatial_dims,
            featmap1_inchn+featmap2_inchn,
            featmap2_inchn,
            kernel_size=1,
            stride=1,
            act='sigmoid',
            norm=None,
            bias=True,
            conv_only=False
        )
        self.w = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)

    def forward(self, x1, x2):
        concat_featmap = torch.cat([x1, x2], dim=1)
        weighted_featmap1 = self.conv1(concat_featmap) * x1
        weighted_featmap2 = self.conv2(concat_featmap) * x2
        # return torch.add(weighted_featmap1, weighted_featmap2)
        return weighted_featmap1 + self.w*weighted_featmap2
