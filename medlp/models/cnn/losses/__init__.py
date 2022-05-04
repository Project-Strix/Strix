import torch
from strix.utilities.registry import Registry
from monai_ex.losses import (
    DiceLoss,
    GeneralizedDiceLoss,
    FocalLoss,
    DiceFocalLoss,
    DiceCELoss,
    DiceTopKLoss
)
from .losses import (
    ContrastiveLoss,
    ContrastiveCELoss,
    ContrastiveBCELoss,
    CrossEntropyLossEx,
    BCEWithLogitsLossEx,
    CombinationLoss,
)

CLASSIFICATION_LOSS = Registry()
SEGMENTATION_LOSS = Registry()
SIAMESE_LOSS = Registry()
SELFLEARNING_LOSS = Registry()
MULTITASK_LOSS = Registry()

LOSS_MAPPING = {
    "classification": CLASSIFICATION_LOSS,
    "segmentation": SEGMENTATION_LOSS,
    "siamese": SIAMESE_LOSS,
    "selflearning": SELFLEARNING_LOSS,
    "multitask": MULTITASK_LOSS,
}

CLASSIFICATION_LOSS.register('CE', CrossEntropyLossEx)
CLASSIFICATION_LOSS.register('WCE', CrossEntropyLossEx)
CLASSIFICATION_LOSS.register('BCE', BCEWithLogitsLossEx)
CLASSIFICATION_LOSS.register('WBCE', BCEWithLogitsLossEx)
# CLASSIFICATION_LOSS.register('FocalLoss', FocalLoss)

SEGMENTATION_LOSS.register('DCE', DiceLoss)
SEGMENTATION_LOSS.register('GDL', GeneralizedDiceLoss)
SEGMENTATION_LOSS.register('CE-DCE', DiceCELoss)
SEGMENTATION_LOSS.register('DiceFocalLoss', DiceFocalLoss)
SEGMENTATION_LOSS.register('DiceTopKLoss', DiceTopKLoss)

SIAMESE_LOSS.register('ContrastiveLoss', ContrastiveLoss)
SIAMESE_LOSS.register('ContrastiveCELoss', ContrastiveCELoss)
SIAMESE_LOSS.register('ContrastiveBCELoss', ContrastiveBCELoss)

SELFLEARNING_LOSS.register('MSE', torch.nn.MSELoss)

MULTITASK_LOSS.register("CombinationLoss", CombinationLoss)
