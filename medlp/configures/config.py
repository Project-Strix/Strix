from pathlib import Path


DEFAULT_MEDLP_CONFIG = {
    "config_fname": 'medlp_configures.cfg',
    "OUTPUT_DIR": str(Path.home()/'Data'/'medlp_exp'),
    "EXTERNAL_DATASET_DIR": str(Path.home()/'Data'/'medlp_datasets'),
}
