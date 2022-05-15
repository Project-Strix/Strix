from pathlib import Path
from strix.utilities.imports import import_file
from strix.configures import config as cfg
from strix.data_io.base_dataset.classification_dataset import *
from strix.data_io.base_dataset.selflearning_dataset import *
from strix.data_io.base_dataset.segmentation_dataset import *
from strix.data_io.base_dataset.siamese_dataset import *

from strix.data_io.dataio import (
    CLASSIFICATION_DATASETS,
    SEGMENTATION_DATASETS,
    SIAMESE_DATASETS,
    SELFLEARNING_DATASETS,
    MULTITASK_DATASETS,
    DATASET_MAPPING,
)
from strix.data_io.generate_dataset import register_dataset_from_cfg

internal_dataset_dir = Path(__file__).parent.parent.joinpath("datasets")
external_dataset_dir = Path(cfg.get_strix_cfg('EXTERNAL_DATASET_DIR'))

dataset_dirs = [internal_dataset_dir, external_dataset_dir]

for data_dir in dataset_dirs:
    if data_dir.is_dir():
        for f in data_dir.glob("*.py"):
            # print('Import module from file:', str(f))
            import_file(f.stem, str(f))
        for j in data_dir.glob("*.yaml"):
            register_dataset_from_cfg(j)
