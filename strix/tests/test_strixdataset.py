import pytest

import json
import torch
import numpy as np

from strix.utilities.registry import DatasetRegistry
from strix.data_io import StrixDataset
from strix.utilities.utils import get_items

from monai_ex.data import Dataset
from monai_ex.transforms import ToTensorD

datalist = [{"image": i, "label": i + 1} for i in range(100)]


@pytest.mark.parametrize("no_transform", [False, True])
def test_strix_dataset(no_transform, tmp_path):
    with open(tmp_path / "datalist.json", "w") as f:
        json.dump(datalist, f, indent=2)

    to_tensor_tf = None if no_transform else ToTensorD(keys=["image", "label"])
    ds_name = f"test-ds-{no_transform}"

    DATASETS = DatasetRegistry()

    @DATASETS.register("2D", "classification", ds_name, str(tmp_path / "datalist.json"))
    def get_dataset(filelist, phase, opts):

        return StrixDataset(
            filelist,
            loader=None,
            channeler=None,
            orienter=None,
            spacer=None,
            rescaler=None,
            resizer=None,
            cropper=None,
            caster=None,
            to_tensor=to_tensor_tf,
            is_supervised=True,
            dataset_type=Dataset,
            dataset_kwargs={},
            check_data=False,
            additional_transforms=[],
        )

    ds_func = DATASETS.get("2D", "classification", ds_name).get("FN")
    fpath = DATASETS.get("2D", "classification", ds_name).get("PATH")
    ds = ds_func(get_items(fpath), "train", {})

    assert ds is not None
    ds_iter = iter(ds)
    item = next(ds_iter)

    if no_transform:
        assert item["image"] == 0 and item["label"] == 1
    else:
        assert item["image"] == torch.tensor([0]) and item["label"] == torch.tensor([1])
