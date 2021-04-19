from pathlib import Path
from medlp.utilities.imports import import_file
from medlp.configures import get_cfg
from medlp.data_io.base_dataset.classification_dataset import *
from medlp.data_io.base_dataset.selflearning_dataset import *
from medlp.data_io.base_dataset.segmentation_dataset import *
from medlp.data_io.base_dataset.siamese_dataset import *

from medlp.data_io.dataio import (
    CLASSIFICATION_DATASETS,
    SEGMENTATION_DATASETS,
    SIAMESE_DATASETS,
    SELFLEARNING_DATASETS,
    MULTITASK_DATASETS,
    DATASET_MAPPING,
)
from medlp.data_io.generate_dataset import register_dataset_from_cfg

internal_dataset_dir = Path(__file__).parent.parent.joinpath("datasets")
external_dataset_dir = Path(get_cfg('MEDLP_CONFIG', 'EXTERNAL_DATASET_DIR'))

dataset_dirs = [internal_dataset_dir, external_dataset_dir]

for data_dir in dataset_dirs:
    if data_dir.is_dir():
        for f in data_dir.glob("*.py"):
            # print('Import module from file:', str(f))
            import_file(f.stem, str(f))
        for j in data_dir.glob("*.yaml"):
            register_dataset_from_cfg(j)
