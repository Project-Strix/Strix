from pathlib import Path
from medlp.utilities.imports import import_file
from medlp.data_io.base_dataset.classification_dataset import *
from medlp.data_io.base_dataset.selflearning_dataset import *
from medlp.data_io.base_dataset.segmentation_dataset import *
from medlp.data_io.dataio import (
    CLASSIFICATION_DATASETS, 
    SEGMENTATION_DATASETS,
    SELFLEARNING_DATASETS,
    MULTITASK_DATASETS,
    DATASET_MAPPING
    )
from medlp.data_io.generate_dataset import register_dataset_config

datasets_dir = Path(__file__).parent.parent.joinpath('datasets')
if datasets_dir.is_dir():
    for f in datasets_dir.glob("*.py"):
        #print('Import module from file:', str(f))
        import_file(f.stem, str(f))
    for j in datasets_dir.glob("*.yaml"):
        register_dataset_config(j)

