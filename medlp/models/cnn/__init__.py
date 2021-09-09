from medlp.utilities.registry import NetworkRegistry

CLASSIFICATION_ARCHI = NetworkRegistry()
SEGMENTATION_ARCHI = NetworkRegistry()
SELFLEARNING_ARCHI = NetworkRegistry()
MULTITASK_ARCHI = NetworkRegistry()
SIAMESE_ARCHI = NetworkRegistry()

ARCHI_MAPPING = {
    "segmentation": SEGMENTATION_ARCHI,
    "classification": CLASSIFICATION_ARCHI,
    "selflearning": SELFLEARNING_ARCHI,
    "multitask": MULTITASK_ARCHI,
    "siamese": SIAMESE_ARCHI,
}

from medlp.models.cnn.medlp_nets import *
