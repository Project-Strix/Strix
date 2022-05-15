import os
import math
import random
import warnings
from typing import Optional, Sequence, Union

from monai_ex.data import Dataset, SplitDataset
from monai_ex.engines.utils import CustomKeys
from monai_ex.transforms import *
from monai_ex.utils import ensure_list
from monai_ex.config import KeysCollection
from numpy.lib.npyio import load
from strix.data_io.base_dataset.classification_dataset import BasicClassificationDataset


class SiameseDatasetWrapper(Dataset):
    """Siamese dataset wrapper.
    Wrap other dataset type such as ClassificationDataset to SiameseDataset.

    """
    def __init__(
        self,
        dataset: Dataset,
        label_key: Optional[KeysCollection] = CustomKeys.LABEL,
        same_ratio: Optional[float] = None,
        max_loop_count: Optional[int] = 10,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.label_key = label_key
        self.same_ratio = same_ratio
        self.max_loop_count = max_loop_count
        if self.same_ratio is not None:
            warnings.warn('Set same ratio may cause inefficient training!')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data1 = self.dataset[idx]
        data2 = None

        if self.same_ratio is None:
            data2 = self.dataset[random.randrange(self.__len__())]
        elif random.random() <= self.same_ratio:
            count = 0
            while count < self.max_loop_count:
                count += 1
                data2 = self.dataset[random.randrange(self.__len__())]
                if data2[self.label_key] == data1[self.label_key]:
                    break
        else:
            count = 0
            while count < self.max_loop_count:
                count += 1
                data2 = self.dataset[random.randrange(self.__len__())]
                if data2[self.label_key] != data1[self.label_key]:
                    break
        assert data2 is not None

        return data1, data2, int(data1[self.label_key] == data2[self.label_key])


class BasicSiameseDataset(BasicClassificationDataset):
    def __new__(
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
    ) -> None:
        super().__init__(
            files_list=files_list,
            loader=loader,
            channeler=channeler,
            orienter=orienter,
            spacer=spacer,
            rescaler=rescaler,
            resizer=resizer,
            cropper=cropper,
            caster=caster,
            to_tensor=to_tensor,
            is_supervised=is_supervised,
            dataset_type=dataset_type,
            dataset_kwargs=dataset_kwargs,
            additional_transforms=additional_transforms,
            verbose=verbose
        )

        assert dataset_type == SplitDataset, f'Only support SplitDataset, but got {dataset_type}'
        # self.label_image_dict = {data[CustomKeys.LABEL]: data for data in self.input_data}
        # self.labels = list(self.label_image_dict.keys())
        return self.dataset(self.input_data, transform=self.transforms, **self.dataset_kwargs)
