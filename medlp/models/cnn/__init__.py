from pathlib import Path
from medlp.utilities.imports import import_file
from medlp.configures import config as cfg
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

external_dataset_dir = Path(cfg.get_medlp_cfg('EXTERNAL_NETWORK_DIR'))
if external_dataset_dir.is_dir():
    for f in external_dataset_dir.glob("*.py"):
        import_file(f.stem, str(f))
