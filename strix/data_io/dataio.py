from typing import List, Optional
from types import SimpleNamespace
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
def get_dataloader(
    args: SimpleNamespace, filelist: List, phase: Phases, unlabel_filelist: Optional[List] = None
):
    """Generate pytorch datalist from given args and filelist.

    Args:
        args (SimpleNamespace): Necessary arguments.
        filelist (List): List containing data file items.
        phase (Phases): Phase of the dataloader
        unlabel_filelist (Optional[List], optional): List containing unlabled data. Defaults to None.

    Raises:
        DatasetException: Exception occured in custom dataset function.
        NotImplementedError: Imbalanced dataset sampling.

    Returns:
        DataLoader: Pytorch's dataloader
    """

    params = get_default_setting(
        phase, train_n_batch=args.n_batch, valid_n_batch=args.n_batch_valid, train_n_workers=args.n_worker
    )  #! How to customize?
    if unlabel_filelist:
        arguments = {"filelist": filelist, "unlabel_filelist": unlabel_filelist, "phase": phase, "opts": vars(args)}
    else:
        arguments = {"filelist": filelist, "phase": phase, "opts": vars(args)}

    try:
        datasets = DatasetRegistry()
        strix_dataset = datasets.get(args.tensor_dim, args.framework, args.data_list)
        torch_dataset = strix_dataset["FN"](**arguments) if strix_dataset is not None else None
    except Exception as e:
        msg = "".join(traceback.format_tb(sys.exc_info()[-1], limit=-1))
        raise DatasetException(f"Dataset {args.data_list} cannot be instantiated!\n{msg}") from e

    label_key = cfg.get_key("LABEL")
    if torch_dataset is None:
        raise DatasetException("None torch dataset is returned from strix_dataset")
    elif isinstance(torch_dataset, _TorchDataLoader):
        return torch_dataset
    elif (
        phase == "train"
        and args.imbalance_sample
        and filelist[0].get(label_key) is not None
    ):
        if isinstance(filelist[0][label_key], (list, tuple)):
            raise NotImplementedError(
                "Imbalanced dataset sampling cannot handle list&tuple label"
            )

        print("Using imbalanced dataset sampling!")
        params.update({"shuffle": False})
        labels = [l[label_key] for l in filelist]

        df = pd.DataFrame()
        df["label"] = labels
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        weights = torch.DoubleTensor(weights.to_list())

        return DataLoader(
            torch_dataset,
            sampler=WeightedRandomSampler(weights=weights, num_samples=len(torch_dataset)),
            **params,
        )
    else:
        return DataLoader(torch_dataset, **params)
