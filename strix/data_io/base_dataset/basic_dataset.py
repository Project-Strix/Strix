from typing import Optional, Sequence, Union

from monai_ex.data import Dataset
from monai_ex.transforms import ComposeEx as Compose
from monai_ex.transforms import MapTransform
from monai_ex.utils import Range, ensure_list

from strix.data_io.base_dataset.utils import get_input_data


class StrixDataset(object):
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
        is_supervised: bool,
        dataset_type: Dataset,
        dataset_kwargs: dict,
        additional_transforms: Optional[Sequence[MapTransform]] = None,
        check_data: bool = True,
        profiling: bool = False,
        verbose: bool = False,
    ):
        self.filelist = filelist
        self.verbose = verbose
        self.dataset = dataset_type
        self.dataset_kwargs = dataset_kwargs
        if check_data:
            self.input_data = get_input_data(filelist, is_supervised, verbose, self.__class__.__name__)
        else:
            self.input_data = filelist

        def _wrap_range(transforms):
            if profiling:
                return [Range(tsf.__class__.__name__)(tsf) for tsf in ensure_list(transforms)]
            else:
                return ensure_list(transforms)

        self.transforms = []
        if loader is not None:
            self.transforms += _wrap_range(loader)
        if channeler is not None:
            self.transforms += _wrap_range(channeler)
        if orienter is not None:
            self.transforms += _wrap_range(orienter)
        if spacer is not None:
            self.transforms += _wrap_range(spacer)
        if rescaler is not None:
            self.transforms += _wrap_range(rescaler)
        if resizer is not None:
            self.transforms += _wrap_range(resizer)
        if cropper is not None:
            self.transforms += _wrap_range(cropper)
        if additional_transforms is not None:
            self.transforms += _wrap_range(additional_transforms)

        if caster is not None:
            self.transforms += _wrap_range(caster)
        if to_tensor is not None:
            self.transforms += _wrap_range(to_tensor)

        if self.transforms:
            self.transforms = Compose(self.transforms)
        else:
            self.transforms = None

        return self.dataset(self.input_data, transform=self.transforms, **self.dataset_kwargs)
