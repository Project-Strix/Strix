import os

import torch
import torch.nn as nn
from medlp.models.cnn.nets.resnet import ResNet, BasicBlock
from medlp.models.cnn.layers.anatomical_gate import AnatomicalAttentionGate as AAG
from medlp.models.cnn.utils import set_trainable
from monai.networks.blocks.dynunet_block import get_conv_layer


class ResNetAAG(ResNet):
    def __init__(
        self, block, layers, dim=2, in_channels=3, roi_classes=1,
        num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64,
        replace_stride_with_dilation=None, norm_layer=None, freeze_backbone=False
    ):
        super().__init__(
            block=block, layers=layers, dim=dim, in_channels=in_channels, num_classes=num_classes,
            zero_init_residual=zero_init_residual, groups=groups, width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation, norm_layer=norm_layer
        )
        self.conv1_img = get_conv_layer(
            dim,
            in_channels,
            64,
            kernel_size=3,
            stride=1,
            act='relu',
            norm='batch',
            bias=True,
            conv_only=False
        )
        roi_chns = [roi_classes, 64, 64, 128, 256]
        self.roi_convs = nn.ModuleList(
            [
                get_conv_layer(
                    dim,
                    chn,
                    roi_chns[i+1],
                    kernel_size=3,
                    stride=1,
                    act='relu',
                    norm='batch',
                    bias=True,
                    conv_only=False
                ) for i, chn in enumerate(roi_chns[:-1])
            ]
        )
        self.aag_layers = nn.ModuleList(
            [AAG(dim, chn, chn) for chn in roi_chns[1:]]
        )

        if freeze_backbone:
            set_trainable(self, False)
            set_trainable(self.fc, True)

    def forward(self, inputs):
        inp, roi = inputs
        x = self.conv1_img(inp)
        y = self.roi_convs[0](roi)
        x = self.maxpool(x)
        y = self.maxpool(y)

        x = self.aag_layers[0](x, y)
        x = self.layer1(x)

        y = self.roi_convs[1](y)
        # y = self.maxpool(y)
        x = self.aag_layers[1](x, y)
        x = self.layer2(x)

        y = self.roi_convs[2](y)
        y = self.maxpool(y)
        x = self.aag_layers[2](x, y)
        x = self.layer3(x)

        y = self.roi_convs[3](y)
        y = self.maxpool(y)
        x = self.aag_layers[3](x, y)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet34_aag(pretrained_model_path, **kwargs):
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        num_classes_ = kwargs['num_classes']
        kwargs['num_classes'] = 1

    model = ResNetAAG(BasicBlock, [3, 4, 6, 3], **kwargs)

    if pretrained_model_path and os.path.exists(pretrained_model_path):
        model.load_state_dict(torch.load(pretrained_model_path))
        in_features_ = model.fc.in_features
        model.fc = nn.Linear(in_features_, num_classes_)

    return model
