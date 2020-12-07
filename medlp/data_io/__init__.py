from pathlib import Path
from medlp.utilities.utils import DatasetRegistry
from medlp.utilities.imports import import_file
from medlp.data_io.base_dataset.classification_dataset import *
from medlp.data_io.base_dataset.selflearning_dataset import *
from medlp.data_io.base_dataset.segmentation_dataset import *


CLASSIFICATION_DATASETS = DatasetRegistry()
SEGMENTATION_DATASETS = DatasetRegistry()
SELFLEARNING_DATASETS = DatasetRegistry()
MULTITASK_DATASETS = DatasetRegistry()

DATASET_MAPPING = {
        'segmentation':SEGMENTATION_DATASETS,
        'classification':CLASSIFICATION_DATASETS,
        'selflearning':SELFLEARNING_DATASETS,
        'multitask':MULTITASK_DATASETS
    }

datasets_dir = Path(__file__).parent.parent.joinpath('datasets')
if datasets_dir.is_dir():
    for f in datasets_dir.glob("*.py"):
        #print('Import module from file:', str(f))
        import_file(f.stem, str(f))


