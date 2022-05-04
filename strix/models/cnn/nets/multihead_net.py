import torch
import torch.nn as nn

from typing import Union


class MultiOutputNet(nn.Module):
    def __init__(
        self,
        backbone: Union[nn.Module, nn.Sequential],
        multiheads: nn.ModuleList,
        mode='independent'
    ):
        super().__init__()
        self.backbone = backbone
        self.multiheads = multiheads
        self.mode = mode
        modes = ['continous', 'independent']
        if self.mode not in modes:
            raise ValueError(f'Mode must be {modes}, but got {self.mode}')

    def forward(self, x):
        x = self.backbone(x)
        if self.mode == 'independent':
            return tuple(head(x) for head in self.multiheads)
        elif self.mode == 'continous':
            outputs = []
            for i, head in enumerate(self.multiheads):
                for h in self.multiheads[:i]:
                    x = h(x)
                outputs.append(head(x))
            return tuple(outputs)
