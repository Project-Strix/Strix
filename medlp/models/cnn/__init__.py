from medlp.utilities.utils import DimRegistry

CLASSIFICATION_ARCHI = DimRegistry()
SEGMENTATION_ARCHI = DimRegistry()
SELFLEARNING_ARCHI = DimRegistry()
MULTITASK_ARCHI = DimRegistry()
SIAMESE_ARCHI = DimRegistry()

ARCHI_MAPPING = {
    "segmentation": SEGMENTATION_ARCHI,
    "classification": CLASSIFICATION_ARCHI,
    "selflearning": SELFLEARNING_ARCHI,
    "multitask": MULTITASK_ARCHI,
    "siamese": SIAMESE_ARCHI,
}

from medlp.models.cnn.nets.dynunet import DynUNet
from medlp.models.cnn.nets.resnet import *
from medlp.models.cnn.nets.scnn import SCNN
from medlp.models.cnn.nets.vgg import *
from medlp.models.cnn.nets.ild_cnn import ILD_Net
from medlp.models.cnn.nets.multihead_net import MultiOutputNet
