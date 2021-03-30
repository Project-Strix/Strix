import torch
from medlp.utilities.utils import Registry
from monai_ex.losses import (
    DiceLoss,
    GeneralizedDiceLoss,
    FocalLoss
)
from .losses import (
    CEDiceLoss,
    FocalDiceLoss,
    ContrastiveLoss,
    ContrastiveCELoss,
    ContrastiveBCELoss,
    CrossEntropyLossEx,
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
CLASSIFICATION_LOSS.register('BCE', torch.nn.BCEWithLogitsLoss)
CLASSIFICATION_LOSS.register('WBCE', torch.nn.BCEWithLogitsLoss)
CLASSIFICATION_LOSS.register('CE-DCE', CEDiceLoss)
CLASSIFICATION_LOSS.register('FocalLoss', FocalLoss)

SEGMENTATION_LOSS.register('DCE', DiceLoss)
SEGMENTATION_LOSS.register('GDL', GeneralizedDiceLoss)
SEGMENTATION_LOSS.register('FocalDiceLoss', FocalDiceLoss)

SIAMESE_LOSS.register('ContrastiveLoss', ContrastiveLoss)
SIAMESE_LOSS.register('ContrastiveCELoss', ContrastiveCELoss)
SIAMESE_LOSS.register('ContrastiveBCELoss', ContrastiveBCELoss)

SELFLEARNING_LOSS.register('MSE', torch.nn.MSELoss)
