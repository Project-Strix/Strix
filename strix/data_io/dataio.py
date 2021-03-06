from multiprocessing.sharedctypes import Value
from typing import Mapping
import sys
import traceback

import torch
from torch.utils.data import DataLoader as _TorchDataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.sampler import WeightedRandomSampler
from strix.utilities.registry import DatasetRegistry
from strix.utilities.enum import Phases
from monai_ex.data import DataLoader
from monai_ex.utils.exceptions import DatasetException
from strix.configures import config as cfg
from strix.utilities.utils import trycatch
import pandas as pd

CLASSIFICATION_DATASETS = DatasetRegistry()
SEGMENTATION_DATASETS = DatasetRegistry()
SELFLEARNING_DATASETS = DatasetRegistry()
MULTITASK_DATASETS = DatasetRegistry()
SIAMESE_DATASETS = DatasetRegistry()

DATASET_MAPPING = {
    "segmentation": SEGMENTATION_DATASETS,
    "classification": CLASSIFICATION_DATASETS,
    "selflearning": SELFLEARNING_DATASETS,
    "multitask": MULTITASK_DATASETS,
    "siamese": SIAMESE_DATASETS,
}


def get_default_setting(phase, **kwargs):
    if phase == Phases.TRAIN:  # Todo: move this part to each dataset?
        shuffle = kwargs.get("train_shuffle", True)
        batch_size = kwargs.get("train_n_batch", 5)
        num_workers = kwargs.get("train_n_workers", 10)
        drop_last = kwargs.get("train_drop_last", True)
        pin_memory = kwargs.get("train_pin_memory", True)
    elif phase == Phases.VALID:
        shuffle = kwargs.get("valid_shuffle", True)
        batch_size = kwargs.get("valid_n_batch", 1)
        num_workers = kwargs.get("valid_n_workers", min(batch_size//2, 1))
        drop_last = kwargs.get("valid_drop_last", False)
        pin_memory = kwargs.get("valid_pin_memory", True)
    elif phase == Phases.TEST_IN or phase == Phases.TEST_EX:
        shuffle = kwargs.get("test_shuffle", False)
        batch_size = kwargs.get("test_n_batch", 1)
        num_workers = kwargs.get("test_n_workers", 1)
        drop_last = kwargs.get("test_drop_last", False)
        pin_memory = kwargs.get("test_pin_memory", True)
    else:
        raise ValueError(f"Phase must be in 'train,valid,test', but got {phase}")

    return {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }


@trycatch()
def get_dataloader(args, files_list, phase):
    params = get_default_setting(
        phase, train_n_batch=args.n_batch, valid_n_batch=args.n_batch_valid, train_n_workers=args.n_worker
    )  #! How to customize?
    arguments = {"files_list": files_list, "phase": phase, "opts": vars(args)}

    try:
        dataset_ = DATASET_MAPPING[args.framework][args.tensor_dim][args.data_list]["FN"](
            **arguments
        )
    except Exception as e:
        msg = "".join(traceback.format_tb(sys.exc_info()[-1], limit=-1))
        raise DatasetException(f"Dataset {args.data_list} cannot be instantiated!\n{msg}") from e

    label_key = cfg.get_key("LABEL")
    if isinstance(dataset_, _TorchDataLoader):
        return dataset_
    elif (
        phase == "train"
        and args.imbalance_sample
        and files_list[0].get(label_key) is not None
    ):
        if isinstance(files_list[0][label_key], (list, tuple)):
            raise NotImplementedError(
                "Imbalanced dataset sampling cannot handle list&tuple label"
            )

        print("Using imbalanced dataset sampling!")
        params.update({"shuffle": False})
        labels = [l[label_key] for l in files_list]

        df = pd.DataFrame()
        df["label"] = labels
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        weights = torch.DoubleTensor(weights.to_list())

        return DataLoader(
            dataset_,
            sampler=WeightedRandomSampler(weights=weights, num_samples=len(dataset_)),
            **params,
        )
    else:
        return DataLoader(dataset_, **params)
