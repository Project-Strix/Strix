import torch
from torch.nn import Module
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Sequence

class ContrastiveLoss(Module):
      def __init__(self, margin=2.0):
            super(ContrastiveLoss, self).__init__()
            self.margin = margin

      def forward(self, output1, output2, label):
            # Find the pairwise distance or eucledian distance of two output feature vectors
            euclidean_distance = F.pairwise_distance(output1, output2)
            # perform contrastive loss calculation with the distance
            loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

            return loss_contrastive

class ContrastiveCELoss(Module):
    def __init__(self, margin=2.0, w=0.5):
        super(ContrastiveCELoss, self).__init__()
        self.margin = margin
        self.ce_loss1 = CrossEntropyLoss()
        self.ce_loss2 = CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss(margin)

    def forward(self, output1, output2, pred_label1, pred_label2, gt1, gt2):
        target = int(gt1 == gt2)
        ce_loss = self.ce_loss1(pred_label1, gt1) + self.ce_loss2(pred_label2, gt2)
        con_loss = self.contrastive_loss(output1, output2, target)
        return ce_loss + con_loss

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

class CEDiceLoss(Module):
    pass