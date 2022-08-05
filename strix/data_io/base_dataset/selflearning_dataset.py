import os
from typing import Optional, Sequence, Union

from monai_ex.data import CacheDataset, PersistentDataset
from monai_ex.data import Dataset
from monai_ex.transforms import *
from monai_ex.utils import ensure_list
from strix.data_io.base_dataset.utils import get_input_data


class BasicSelflearningDataset(object):
    def __new__(
        self,
        filelist: Sequence,
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
        additional_transforms: Optional[Sequence[MapTransform]] = None,
        check_data: bool = True,
        verbose: bool = False,
    ):
        self.filelist = filelist
        self.verbose = verbose
        self.dataset = dataset_type
        self.dataset_kwargs = dataset_kwargs
        if check_data:
            self.input_data = get_input_data(
                filelist, False, verbose, self.__class__.__name__
            )
        else:
            self.input_data = filelist

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

        return self.dataset(self.input_data, transform=self.transforms, **self.dataset_kwargs)

