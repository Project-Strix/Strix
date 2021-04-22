import os
from typing import Optional, Sequence, Union

from monai_ex.data import CacheDataset, PersistentDataset
from monai_ex.data import Dataset
from monai_ex.transforms import *
from monai_ex.utils import ensure_list


class BasicSelflearningDataset(object):
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
        dataset_type: Dataset,
        dataset_kwargs: dict,
        additional_transforms,
        verbose=False,
    ):
        self.files_list = files_list
        self.verbose = verbose
        self.dataset = dataset_type
        self.dataset_kwargs = dataset_kwargs
        self.input_data = self.get_input_data()

        self.transforms = ensure_list(loader)
        if channeler is not None:
            self.transforms += ensure_list(channeler)
        if orienter is not None:
            self.transforms += ensure_list(orienter)
        if spacer is not None:
            self.transforms += ensure_list(spacer)
        if rescaler is not None:
            self.transforms += ensure_list(rescaler)
        if additional_transforms is not None:
            self.transforms += ensure_list(additional_transforms)
        if resizer is not None:
            self.transforms += ensure_list(resizer)
        if cropper is not None:
            self.transforms += ensure_list(cropper)

        self.transforms = Compose(self.transforms)

    def get_input_data(self):
        """
        check input file_list format and existence.
        """
        input_data = []
        for f in self.files_list:
            if isinstance(f, str):  # Recognize f as a file
                assert os.path.exists(f), f"File not exists: {f}"
                input_data.append({"image": f, "label": f})
            elif isinstance(f, dict):
                assert "image" in f, f"File {f} doesn't contain keyword 'image'"
                assert os.path.exists(f["image"]), f"File not exists: {f['image']}"
                input_data.append({"image": f["image"], "label": f["image"]})
        return input_data

    def get_dataset(self):
        return self.dataset(self.input_data, transform=self.transforms, **self.dataset_kwargs)

