import os
import numpy as np
from typing import Optional, Sequence, Union

# from monai_ex.engines.utils import CustomKeys, get_keys_list
from monai_ex.data import Dataset
from monai_ex.transforms import *
from monai_ex.utils import ensure_list


class BasicSegmentationDataset(object):
    def __init__(
        self,
        files_list: Sequence,
        loader: Union[Sequence[MapTransform], MapTransform],
        channeler: Union[Sequence[MapTransform], MapTransform],
        orienter: Union[Sequence[MapTransform], MapTransform],
        spacer: Union[Sequence[MapTransform], MapTransform],
        rescaler: Union[Sequence[MapTransform], MapTransform],
        resizer: Union[Sequence[MapTransform], MapTransform],
        cropper: Union[Sequence[MapTransform], MapTransform],
        caster: Union[Sequence[MapTransform], MapTransform],
        to_tensor: Union[Sequence[MapTransform], MapTransform],
        is_supervised: bool,
        dataset_type: Dataset,
        dataset_kwargs: dict,
        additional_transforms: Optional[Sequence[MapTransform]] = None,
        verbose: Optional[bool] = False,
    ):
        self.files_list = files_list
        self.verbose = verbose
        self.dataset = dataset_type
        self.dataset_kwargs = dataset_kwargs
        self.input_data = self.get_input_data(is_supervised)

        self.transforms = ensure_list(loader)
        if channeler is not None:
            self.transforms += ensure_list(channeler)
        if orienter is not None:
            self.transforms += ensure_list(orienter)
        if spacer is not None:
            self.transforms += ensure_list(spacer)
        if rescaler is not None:
            self.transforms += ensure_list(rescaler)
        if resizer is not None:
            self.transforms += ensure_list(resizer)
        if cropper is not None:
            self.transforms += ensure_list(cropper)
        if additional_transforms is not None:
            self.transforms += ensure_list(additional_transforms)

        if caster is not None:
            self.transforms += ensure_list(caster)
        if to_tensor is not None:
            self.transforms += ensure_list(to_tensor)

        self.transforms = Compose(self.transforms)

    def get_input_data(self, is_supervised):
        """
        check input file_list format and existence.
        """
        input_data = []
        for f in self.files_list:
            if is_supervised:
                if isinstance(f, (list, tuple)):  # Recognize f as ['image','label']
                    assert os.path.exists(f[0]), f"Image file not exists: {f[0]}"
                    assert os.path.exists(f[1]), f"Image file not exists: {f[1]}"
                    input_data.append({"image": f[0], "label": f[1]})
                elif isinstance(f, dict):
                    assert "image" in f, f"File {f} doesn't contain keyword 'image'"
                    assert os.path.exists(f["image"]), f"File not exists: {f['image']}"
                    input_data.append({"image": f["image"], "label": f["label"]})
                else:
                    raise ValueError(
                        f"Not supported file_list format, Got {type(f)} in SupervisedDataset"
                    )
            else:
                if isinstance(f, str):
                    assert os.path.exists(f), f"Image file not exists: {f}"
                    input_data.append({"image": f})
                elif isinstance(f, dict):
                    assert "image" in f, f"File {f} doesn't contain keyword 'image'"
                    assert os.path.exists(f["image"]), f"File not exists: {f['image']}"
                    input_data.append({"image": f["image"]})
                else:
                    raise ValueError(
                        f"Not supported file_list format, Got {type(f)} in UnsupervisedDataset"
                    )
        return input_data

    def get_dataset(self):
        return self.dataset(self.input_data, transform=self.transforms, **self.dataset_kwargs)
