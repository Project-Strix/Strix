import torch
import torch.nn as nn
from typing import Callable
import numpy as np
from medlp.models.cnn import CLASSIFICATION_ARCHI
from monai_ex.networks.layers import Conv, Dropout, Norm, Pool, PrunableLinear


def get_FeatureMaps(L, policy, constant=17):
    return {
        'proportional': (L+1)**2,
        'static': constant,
    }[policy]


@CLASSIFICATION_ARCHI.register('2D', 'ild_net')
@CLASSIFICATION_ARCHI.register('3D', 'ild_net')
class ILD_Net(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_depth=5,
        k=4,
        feature_policy='proportional',
        init_weights=True,
        **kwargs
    ):
        super(ILD_Net, self).__init__()

        dim = kwargs.get('dim', 2)
        is_prunable = kwargs.get('is_prunable', False)
        bottleneck_size = 1  # kwargs.get('bottleneck_size', 1)
        output_size = (bottleneck_size, )*dim
        num_ = np.prod(output_size)

        conv_type: Callable = Conv[Conv.CONV, dim] if not is_prunable else Conv[Conv.PRUNABLE_CONV, dim]
        norm_type: Callable = Norm[Norm.BATCH, dim]
        pool_type: Callable = Pool[Pool.ADAPTIVEAVG, dim]
        fc_type: Callable = nn.Linear if not is_prunable else PrunableLinear
        fc_unit = (
            int(k*get_FeatureMaps(n_depth, feature_policy))//6,
            int(k*get_FeatureMaps(n_depth, feature_policy))//2
        )
        self.avgpool = pool_type(output_size)

        layers = []
        layers.append(conv_type(in_channels, k*get_FeatureMaps(1, feature_policy), kernel_size=2))
        layers.append(nn.LeakyReLU(0.3))
        for i in range(2, n_depth+1):
            layers.append(conv_type(
                k*get_FeatureMaps(i-1, feature_policy),
                k*get_FeatureMaps(i, feature_policy),
                kernel_size=2
            ))
            layers.append(nn.LeakyReLU(0.3))
        self.features = nn.Sequential(*layers)

        self.classifier = nn.Sequential(
            fc_type(int(k*get_FeatureMaps(n_depth, feature_policy))*num_, fc_unit[0]),
            nn.LeakyReLU(0, True),
            nn.Dropout(0.5),
            fc_type(fc_unit[0], fc_unit[1]),
            nn.LeakyReLU(0, True),
            nn.Dropout(0.5),
            fc_type(fc_unit[1], out_channels),
        )

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
