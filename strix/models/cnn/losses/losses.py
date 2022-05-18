from typing import Callable, Optional, Sequence, Union

import torch
import torch.nn.functional as F
from monai_ex.losses import DiceLoss
from torch import Tensor
from torch.nn import Module
from monai_ex.utils import ensure_same_dim

# class ContrastiveLoss(Module):
#     def __init__(self, margin=2.0):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin

#     def forward(self, output1, output2, target1, target2, size_average=True):
#         label = (~target1.eq(target2)).to(torch.int16)
#         # Find the pairwise distance or eucledian distance of two output feature vectors
#         euclidean_distance = F.pairwise_distance(output1, output2)
#         # perform contrastive loss calculation with the distance
#         loss_contrastive = (1-label) * torch.pow(euclidean_distance, 2) + \
#                            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
#         return torch.mean(loss_contrastive) if size_average else torch.sum(loss_contrastive)


class CrossEntropyLossEx(torch.nn.CrossEntropyLoss):
    """Extension of pytorch's CrossEntropyLoss
       Enable passing float type weight argument.
    """
    def __init__(
        self,
        weight: Optional[Union[torch.Tensor, float]] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = 'mean',
        device: Optional[torch.device] = None
    ) -> None:
        if weight is not None:
            weight = torch.tensor(weight)
            if device is not None:
                weight = weight.to(device)

        super().__init__(
            weight=weight,
            size_average=size_average,
            ignore_index=ignore_index,
            reduce=reduce,
            reduction=reduction
        )


class BCEWithLogitsLossEx(torch.nn.BCEWithLogitsLoss):
    """Extension of pytorch's BCEWithLogitsLoss
       Enable passing float type weight&pos_weight argument

    Args:
        torch ([type]): [description]
    """
    def __init__(
        self,
        weight: Optional[Union[torch.Tensor, float]] = None,
        size_average=None,
        reduce=None,
        reduction: str = 'mean',
        pos_weight: Optional[Union[torch.Tensor, float]] = None,
        device: Optional[torch.device] = None
    ) -> None:
        if weight is not None:
            weight = torch.tensor(weight)
            if device is not None:
                weight = weight.to(device)
        if pos_weight is not None:
            pos_weight = torch.tensor(pos_weight)
            if device is not None:
                pos_weight = pos_weight.to(device)

        super().__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )


class ContrastiveLoss(Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin=10):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target1, target2, size_average=True):
        target = target1.eq(target2).to(torch.int16)
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances +
                        (1 + -1 * target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class ContrastiveCELoss(Module):
    def __init__(self, margin=2.0, reduction="mean"):
        super(ContrastiveCELoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        assert self.reduction in ['sum', 'mean'], f"reduction must be 'sum' or 'mean', but got {reduction}"
        self.ce_loss = torch.nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss(margin)

    def forward(self, output1, output2, pred1, pred2, target1, target2):
        ce_loss = self.ce_loss(pred1, target1) + self.ce_loss(pred2, target2)
        con_loss = self.contrastive_loss(output1, output2, target1, target2)
        return ce_loss+con_loss if self.reduction == 'sum' else (ce_loss+con_loss)/2


class ContrastiveBCELoss(Module):
    def __init__(self, margin=2.0, reduction="sum"):
        super(ContrastiveBCELoss, self).__init__()
        self.margin = margin
        self.reduction = reduction
        assert self.reduction in ['sum', 'mean'], f"reduction must be 'sum' or 'mean', but got {reduction}"
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.contrastive_loss = ContrastiveLoss(margin)

    def forward(self, output1, output2, pred1, pred2, target1, target2):
        bce_loss = self.bce_loss(pred1, target1) + self.bce_loss(pred2, target2)
        con_loss = self.contrastive_loss(output1, output2, target1, target2)
        return bce_loss+con_loss if self.reduction == 'sum' else (bce_loss+con_loss)/2


class DeepSupervisionLoss(Module):
    def __init__(self, base_loss):
        super(DeepSupervisionLoss, self).__init__()
        self.base_loss = base_loss

    def forward(self, inputs: Sequence[Tensor], gt: Tensor):
        deep_sup_num = len(inputs)-1
        weights = [0.5]+[0.5/deep_sup_num]*deep_sup_num
        losses = []
        for w, ret in zip(weights, inputs):
            losses.append(w*self.base_loss(ret, gt))
        return torch.mean(torch.stack(losses))


class CombinationLoss(Module):
    def __init__(self, loss1: Callable, loss2: Callable, aggregate: str = "sum", weight: float = 1.0, ensure_dim: bool = True):
        super(CombinationLoss, self).__init__()
        self.aggregate = aggregate
        self.ensure_dim = ensure_dim
        self.loss1 = loss1
        self.loss2 = loss2
        self.weight = weight

    def forward(self, net_output, target):
        if not isinstance(net_output, tuple) or len(net_output) != 2:
            raise ValueError(
                "The network output must be tuple w/ size of 2, "
                f"but got {type(net_output)} w/ size of {len(net_output)}."
            )

        if self.ensure_dim:
            loss1_output = self.loss1(*ensure_same_dim(net_output[0], target[0]))
            loss2_output = self.loss2(*ensure_same_dim(net_output[1], target[1]))
        else:
            loss1_output = self.loss1(net_output[0], target[0])
            loss2_output = self.loss2(net_output[1], target[1])

        if self.aggregate == "sum":
            result = self.weight * loss1_output + loss2_output
        else:
            raise NotImplementedError
        return result
