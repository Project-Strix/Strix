from pathlib import Path
from medlp.utilities.utils import DimRegistry
from medlp.utilities.imports import import_file
from medlp.data_io.base_dataset.classification_dataset import *
from medlp.data_io.base_dataset.selflearning_dataset import *
from medlp.data_io.base_dataset.segmentation_dataset import *


CLASSIFICATION_DATASETS = DimRegistry()
SEGMENTATION_DATASETS = DimRegistry()
SELFLEARNING_DATASETS = DimRegistry()
MULTITASK_DATASETS = DimRegistry()

datasets_dir = Path(__file__).parent.parent.joinpath('datasets')
if datasets_dir.is_dir():
    for f in datasets_dir.glob("*.py"):
        #print('Import module from file:', str(f))
        import_file(f.stem, str(f))


