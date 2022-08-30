import torch
from strix.utilities.registry import LossRegistry
from monai_ex.losses import DiceLoss, GeneralizedDiceLoss, FocalLoss, DiceFocalLoss, DiceCELoss, DiceTopKLoss
from .losses import (
    ContrastiveLoss,
    ContrastiveCELoss,
    ContrastiveBCELoss,
    CrossEntropyLossEx,
    BCEWithLogitsLossEx,
    CombinationLoss,
)

losses = LossRegistry()

losses.register("classification", "CE", CrossEntropyLossEx)
losses.register("classification", "WCE", CrossEntropyLossEx)
losses.register("classification", "BCE", BCEWithLogitsLossEx)
losses.register("classification", "WBCE", BCEWithLogitsLossEx)
# CLASSIFICATION_LOSS.register('FocalLoss', FocalLoss)

losses.register("segmentation", "DCE", DiceLoss)
losses.register("segmentation", "GDL", GeneralizedDiceLoss)
losses.register("segmentation", "CE-DCE", DiceCELoss)
losses.register("segmentation", "DiceFocalLoss", DiceFocalLoss)
losses.register("segmentation", "DiceTopKLoss", DiceTopKLoss)

# SIAMESE_LOSS.register('ContrastiveLoss', ContrastiveLoss)
# SIAMESE_LOSS.register('ContrastiveCELoss', ContrastiveCELoss)
# SIAMESE_LOSS.register('ContrastiveBCELoss', ContrastiveBCELoss)

losses.register("selflearning", "MSE", torch.nn.MSELoss)

losses.register("multitask", "Uniform", CombinationLoss)
losses.register("multitask", "Weighted", CombinationLoss)
