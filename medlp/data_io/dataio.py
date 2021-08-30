import torch
from torch.utils.data import DataLoader as _TorchDataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from medlp.utilities.utils import DatasetRegistry
from monai_ex.data import DataLoader
from medlp.configures import config as cfg
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
    if phase == "train":  # Todo: move this part to each dataset
        shuffle = kwargs.get("train_shuffle", True)
        batch_size = kwargs.get("train_n_batch", 5)
        num_workers = kwargs.get("train_n_workers", 10)
        drop_last = kwargs.get("train_drop_last", True)
        pin_memory = kwargs.get("train_pin_memory", True)
    elif phase == "valid":
        shuffle = kwargs.get("valid_shuffle", True)
        batch_size = kwargs.get("valid_n_batch", 2)
        num_workers = kwargs.get("valid_n_workers", 2)
        drop_last = kwargs.get("valid_drop_last", False)
        pin_memory = kwargs.get("valid_pin_memory", True)
    elif phase == "test" or phase == "test_wo_label":
        shuffle = kwargs.get("test_shuffle", False)
        batch_size = kwargs.get("test_n_batch", 1)
        num_workers = kwargs.get("test_n_workers", 2)
        drop_last = kwargs.get("test_drop_last", False)
        pin_memory = kwargs.get("test_pin_memory", True)
    else:
        raise ValueError(f"phase must be in 'train,valid,test', but got {phase}")

    return {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }


def get_dataloader(args, files_list, phase="train"):
    params = get_default_setting(
        phase, train_n_batch=args.n_batch, valid_n_batch=4
    )  #! How to customize?
    arguments = {"files_list": files_list, "phase": phase, "opts": vars(args)}

    dataset_ = DATASET_MAPPING[args.framework][args.tensor_dim][args.data_list]['FN'](
        **arguments
    )

    label_key = cfg.get_key("LABEL")
    if isinstance(dataset_, _TorchDataLoader):
        return dataset_
    elif phase=="train" and args.imbalance_sample and files_list[0].get(label_key) is not None:
        if isinstance(files_list[0][label_key], (list, tuple)):
            raise NotImplementedError('Imbalanced dataset sampling cannot handle list&tuple label')

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
            **params
        )
    else:
        return DataLoader(dataset_, **params)
