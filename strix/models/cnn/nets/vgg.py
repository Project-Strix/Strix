from typing import Callable, Any

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
import numpy as np

from monai_ex.networks.layers import Conv, Norm, Pool, PrunableLinear

__all__ = [
    "VGG",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
]


model_urls = {
    "vgg11": "https://download.pytorch.org/models/vgg11-bbd30ac9.pth",
    "vgg13": "https://download.pytorch.org/models/vgg13-c768596a.pth",
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
    "vgg19": "https://download.pytorch.org/models/vgg19-dcbb9e9d.pth",
    "vgg11_bn": "https://download.pytorch.org/models/vgg11_bn-6002323d.pth",
    "vgg13_bn": "https://download.pytorch.org/models/vgg13_bn-abd245e5.pth",
    "vgg16_bn": "https://download.pytorch.org/models/vgg16_bn-6c64b313.pth",
    "vgg19_bn": "https://download.pytorch.org/models/vgg19_bn-c79401a0.pth",
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True, **kwargs):
        super(VGG, self).__init__()
        dim = kwargs.get("dim", 2)
        is_prunable = kwargs.get("is_prunable", False)
        bottleneck_size = kwargs.get("bottleneck_size", 7)
        n_group = 1  # kwargs.get('n_group', 1)

        pool_type: Callable = Pool[Pool.ADAPTIVEAVG, dim]
        fc_type: Callable = nn.Linear if not is_prunable else PrunableLinear

        self.features = features
        output_size = (int(bottleneck_size),) * dim  # (2,2,2) #For OOM issue

        self.avgpool = pool_type(output_size)
        num_ = np.prod(output_size).astype(int)
        self.classifier = nn.Sequential(
            fc_type(512 * num_ * n_group, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            fc_type(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            fc_type(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class VGG_MultiOut(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True, **kwargs):
        super().__init__()
        dim = kwargs.get("dim", 2)
        is_prunable = kwargs.get("is_prunable", False)
        bottleneck_size = kwargs.get("bottleneck_size", 7)
        self.latent_only = kwargs.get("siamese") == "single"

        pool_type: Callable = Pool[Pool.ADAPTIVEAVG, dim]
        fc_type: Callable = nn.Linear if not is_prunable else PrunableLinear

        self.features = features
        output_size = (bottleneck_size,) * dim  # (2,2,2) #For OOM issue

        self.avgpool = pool_type(output_size)
        num_ = np.prod(output_size)
        self.embedder = nn.Identity()
        self.classifier = nn.Sequential(
            fc_type(512 * num_, 256 * num_),
            nn.ReLU(True),
            nn.Dropout(0),
            fc_type(256 * num_, 128 * num_),
            nn.ReLU(True),
            nn.Dropout(0),
            fc_type(128 * num_, num_classes),
        )
        if init_weights:
            self.initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if self.latent_only:
            return self.embedder(x)
        else:
            return self.embedder(x), self.classifier(x)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(
    cfg,
    dim,
    in_channels=3,
    batch_norm=False,
    is_prunable=False,
    n_group=1,
):
    layers = []
    conv_type: Callable = (
        Conv[Conv.CONV, dim] if not is_prunable else Conv[Conv.PRUNABLE_CONV, dim]
    )
    norm_type: Callable = Norm[Norm.BATCH, dim]
    pool_type: Callable = Pool[Pool.MAX, dim]
    input_channels = in_channels
    for idx, v in enumerate(cfg):
        if v == "M":
            layers += [pool_type(kernel_size=2, stride=2)]
        else:
            if idx == len(cfg) - 1:
                n_group = 1

            v *= n_group
            conv = conv_type(
                input_channels, v, kernel_size=3, padding=1, groups=n_group
            )
            if batch_norm:
                layers += [conv, norm_type(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            input_channels = v
    return nn.Sequential(*layers)


cfgs = {
    "S": [64, 128, "M", 256, 256, "M", 512, 512],
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M" ,128, 128, "M" ,256, 256, 256, "M" ,512, 512, 512, "M" ,512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs["init_weights"] = False

    in_channels = kwargs.get("in_channels", 3)
    dim = kwargs.get("dim", 2)
    is_prunable = kwargs.get("is_prunable", False)

    if pretrained:
        num_classes_ = kwargs["num_classes"]
        kwargs["num_classes"] = 1000

    if kwargs.get("siamese", None):
        model = VGG_MultiOut(
            make_layers(
                cfgs[cfg],
                dim,
                in_channels=in_channels,
                batch_norm=batch_norm,
                is_prunable=is_prunable,
            ),
            **kwargs
        )
    else:
        model = VGG(
            make_layers(
                cfgs[cfg],
                dim,
                in_channels=in_channels,
                batch_norm=batch_norm,
                is_prunable=is_prunable,
                n_group=kwargs.get("n_group", 1),
            ),
            **kwargs
        )

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)

        classifier_ = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, num_classes_),
        )

        model.classifier = classifier_

    return model


def vgg9(pretrained=False, progress=True, **kwargs):
    r"""VGG 9-layer model (configuration "S") for small size dataset
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "S", False, pretrained, progress, **kwargs)


def vgg9_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 9-layer model (configuration "S") with batch normalization for small size dataset
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "S", True, pretrained, progress, **kwargs)


def vgg11(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, pretrained, progress, **kwargs)


def vgg11_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, pretrained, progress, **kwargs)


def vgg13(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", False, pretrained, progress, **kwargs)


def vgg13_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, pretrained, progress, **kwargs)


def vgg16(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)


def vgg16_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16_bn", "D", True, pretrained, progress, **kwargs)


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False, pretrained, progress, **kwargs)


def vgg19_bn(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, pretrained, progress, **kwargs)
