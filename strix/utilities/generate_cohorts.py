from typing import Optional, List
import os
import logging
from pathlib import Path
from shutil import copyfile
from strix.configures import config as cfg
from strix.utilities.registry import DatasetRegistry
from strix.utilities.utils import get_items, generate_synthetic_datalist
from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from utils_cw import split_train_test
import numpy as np


def generate_train_valid_cohorts(
    tensor_dim: int,
    framework: str,
    data_list: str,
    experiment_path: Path,
    split: float,
    seed: int,
    partial: float,
    n_fold: int,
    n_repeat: int,
    do_test: bool = False,
    ith_fold: int = 0,
    train_list: Optional[str] = None,
    valid_list: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    **kwargs,
):
    datasets = DatasetRegistry()
    strix_dataset = datasets.get(tensor_dim, framework, data_list)
    if strix_dataset is None:
        raise ValueError(f"Dataset {data_list} not found!")

    source_file = strix_dataset.get("SOURCE")
    if source_file and os.path.isfile(source_file):
        copyfile(source_file, experiment_path.joinpath(f"{data_list}.snapshot"))

    # ! Manually specified train&valid datalist
    if train_list and valid_list:
        files_train = get_items(train_list, format="auto")
        files_valid = get_items(valid_list, format="auto")
        return [(files_train, files_valid), ]

    datalist_fpath = strix_dataset.get("PATH", "")
    testlist_fpath = strix_dataset.get("TEST_PATH")

    # ! Synthetic test phase
    if datalist_fpath is None:
        train_datalist = generate_synthetic_datalist(100, logger)
    else:
        assert os.path.isfile(datalist_fpath), f"Data list '{datalist_fpath}' not exists!"
        train_datalist = get_items(datalist_fpath, format="auto")

    if not train_datalist:
        raise ValueError("No train datalist if found!")

    if do_test and (testlist_fpath is None or not os.path.isfile(testlist_fpath)):
        if logger:
            logger.warn(
                f"Test datalist is not found, split test cohort from training data with split ratio of {split}"
            )
        train_test_cohort = split_train_test(
            train_datalist, split, cfg.get_key("label"), 1, random_seed=seed
        )
        train_datalist, _ = train_test_cohort[0]

    if 0 < partial < 1:
        if logger:
            logger.info("Use {} data".format(int(len(train_datalist) * partial)))
        train_datalist = train_datalist[: int(len(train_datalist) * partial)]
    elif partial > 1 or partial == 0:
        if logger:
            logger.warn(f"Expect 0 < partial < 1, but got {partial}. Ignored.")

    split = int(split) if split >= 1 else split
    cohorts = []
    if n_fold > 1 or n_repeat > 1:  # ! K-fold cross-validation
        if n_fold > 1:
            kf = KFold(n_splits=n_fold, random_state=seed, shuffle=True)
        elif n_repeat > 1:
            kf = ShuffleSplit(n_splits=n_repeat, test_size=split, random_state=seed)
        else:
            raise ValueError(f"Got unexpected n_fold({n_fold}) or n_repeat({n_repeat})")

        for i, (train_index, test_index) in enumerate(kf.split(train_datalist)):
            ith = i if ith_fold < 0 else ith_fold
            if i < ith:
                continue
            train_data = list(np.array(train_datalist)[train_index])
            valid_data = list(np.array(train_datalist)[test_index])
            cohorts.append((train_data, valid_data))

    else:  # ! Plain training
        train_data, valid_data = train_test_split(train_datalist, test_size=split, random_state=seed)
        cohorts.append((train_data, valid_data))

    return cohorts


def generate_test_cohort(
    tensor_dim: int,
    framework: str,
    data_list: str,
    do_test: bool,
    split: float,
    seed: int,
) -> Optional[List]:
    """Generate test cohort.

    Args:
        tensor_dim (int): tensor dim
        framework (str): framework.
        data_list (str): datalist name.
        do_test (bool): train_and_test flag.
        split (float): split ratio.
        seed (int): random seed for train_test_split.

    Raises:
        ValueError: dataset is not found.
        ValueError: no test file is found and train_datalist is not given.

    Returns:
        Optional[List]: _description_
    """
    datasets = DatasetRegistry()
    strix_dataset = datasets.get(tensor_dim, framework, data_list)
    if strix_dataset is None:
        raise ValueError(f"Dataset {data_list} not found!")

    testlist_fpath = strix_dataset.get("TEST_PATH")
    if testlist_fpath and os.path.isfile(testlist_fpath):
        test_datalist = get_items(testlist_fpath, format="auto")
    elif do_test:
        datalist_fpath = strix_dataset.get("PATH", "")

        if datalist_fpath is None:
            train_datalist = generate_synthetic_datalist(100, None)
        else:
            assert os.path.isfile(datalist_fpath), f"Data list '{datalist_fpath}' not exists!"
            train_datalist = get_items(datalist_fpath, format="auto")

        if not train_datalist:
            raise ValueError("No train datalist if found!")

        train_test_cohort = split_train_test(
            train_datalist, split, cfg.get_key("label"), 1, random_seed=seed
        )
        _, test_datalist = train_test_cohort[0]
    else:
        raise ValueError("No test file is given and do_test flag is off! Cannot generate test cohort.")

    return test_datalist
