import pytest
import warnings
from types import SimpleNamespace as sn
import random

import torch

from strix import strix_networks, strix_datasets
from strix.data_io.dataio import get_dataloader
from strix.utilities.enum import Phases
from strix.utilities.utils import generate_synthetic_datalist
from monai_ex.utils.exceptions import DatasetException


def test_weigthsampling_str_label_error():
    opts1 = {
        "tensor_dim": "2D", "framework": "classification", "data_list": "RandomData", "output_nc": 3, "save_n_best": 1, "save_epoch_freq": 1, "nni": False, "n_batch": 10,
        "n_batch_valid": 2, "n_worker": 0, "n_epoch": 1, "early_stop": 0, "n_epoch_len": 1, "visualize": False, "criterion": "CE", "imbalance_sample": True
    }
    opts = sn(**opts1)

    filelist = generate_synthetic_datalist(200)
    with pytest.raises(DatasetException):
        dataloader = get_dataloader.__wrapped__(opts, filelist, Phases.TRAIN)  # remove decorator


def test_weigthsampling():
    data_num = 300
    opts1 = {
        "tensor_dim": "2D", "framework": "classification", "data_list": "RandomData", "output_nc": 3, "save_n_best": 1, "save_epoch_freq": 1, "nni": False, "n_batch": 10,
        "n_batch_valid": 2, "n_worker": 0, "n_epoch": 1, "early_stop": 0, "n_epoch_len": 1, "visualize": False, "criterion": "CE", "imbalance_sample": True
    }
    opts = sn(**opts1)

    filelist = [
        {"image": f"synthetic_image{i}.nii.gz", "label": random.randint(0, opts.output_nc)} for i in range(data_num)
    ]
    dataloader = get_dataloader(opts, filelist, Phases.TRAIN)
    
    assert isinstance(dataloader.sampler.weights, torch.Tensor)
    assert dataloader.sampler.num_samples == data_num
