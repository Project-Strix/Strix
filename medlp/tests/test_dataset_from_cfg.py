from pathlib import Path
from strix.data_io.generate_dataset import test_dataset_from_config


fname = Path("/homes/clwang/test_config.yaml")
test_dataset_from_config(fname, 'train', None)
