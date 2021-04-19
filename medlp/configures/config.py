from pathlib import Path


DEFAULT_MEDLP_CONFIG = {
    "CONFIG_FNAME": 'medlp_configures.cfg',
    "OUTPUT_DIR": str(Path.home()/'Data'/'medlp_exp'),
    "EXTERNAL_DATASET_DIR": str(Path.home()/'Data'/'medlp_datasets'),
}
