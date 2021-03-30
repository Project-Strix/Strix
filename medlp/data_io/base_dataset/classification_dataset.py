import os
import math
import numpy as np
from typing import Optional, Sequence, Union

from monai_ex.data import Dataset, CacheDataset, PersistentDataset
from monai_ex.engines.utils import CustomKeys, get_keys_list
from monai_ex.transforms import *
from monai_ex.utils import ensure_list
from medlp.data_io.base_dataset.utils import get_input_data


class BasicClassificationDataset(object):
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
        self.input_data = get_input_data(
            files_list, is_supervised, verbose, self.__class__.__name__
        )

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

    def get_dataset(self):
        return self.dataset(self.input_data, transform=self.transforms, **self.dataset_kwargs)


# class SupervisedClassificationDataset2D(BasicClassificationDataset):
#     def __init__(
#         self,
#         files_list,
#         loader: MapTransform = LoadPNGExD(keys=CustomKeys.IMAGE, grayscale=True),
#         channeler: Optional[MapTransform] = AddChannelD(keys=CustomKeys.IMAGE),
#         orienter: Optional[MapTransform] = OrientationD(keys=CustomKeys.IMAGE, axcodes="LPI"),
#         spacer: Optional[MapTransform] = SpacingD(keys=CustomKeys.IMAGE, pixdim=(0.1, 0.1)),
#         rescaler: Optional[MapTransform] = ScaleIntensityRangeD(
#             keys=CustomKeys.IMAGE, a_min=0, a_max=255, b_min=0, b_max=1, clip=True
#         ),
#         resizer: Optional[MapTransform] = ResizeD(
#             keys=CustomKeys.IMAGE, spatial_size=(512, 512)
#         ),
#         cropper: Optional[MapTransform] = RandSpatialCropD(
#             keys=CustomKeys.IMAGE, roi_size=(256, 256), random_size=False
#         ),
#         caster: Optional[MapTransform] = CastToTypeD(
#             keys=[CustomKeys.IMAGE, CustomKeys.LABEL], dtype=[np.float32, np.int64]
#         ),
#         to_tensor: Optional[MapTransform] = ToTensorD(keys=[CustomKeys.IMAGE, CustomKeys.LABEL]),
#         dataset_type: Dataset = CacheDataset,
#         dataset_kwargs: dict = {'cache_rate': 1.0},
#         additional_transforms: Optional[Sequence] = None,
#         verbose: bool = False,
#     ):
#         assert loader is not None, "Loader cannot be None"

#         super().__init__(
#             files_list,
#             loader,
#             channeler,
#             orienter,
#             spacer,
#             rescaler,
#             resizer,
#             cropper,
#             caster,
#             to_tensor,
#             True,
#             dataset_type,
#             dataset_kwargs,
#             additional_transforms,
#             verbose,
#         )


# class SupervisedClassificationDataset(BasicClassificationDataset):
#     def __init__(
#         self,
#         files_list,
#         loader: MapTransform = LoadNiftid(
#             keys=CustomKeys.IMAGE, dtype=np.float32
#         ),
#         channeler: Optional[MapTransform] = AddChanneld(
#             keys=CustomKeys.IMAGE
#         ),
#         orienter: Optional[MapTransform] = Orientationd(
#             keys=CustomKeys.IMAGE, axcodes="LPI"
#         ),
#         spacer: Optional[MapTransform] = Spacingd(
#             keys=CustomKeys.IMAGE, pixdim=(1, 1, 1)
#         ),
#         rescaler: Optional[MapTransform] = NormalizeIntensityD(
#             keys=CustomKeys.IMAGE
#         ),
#         resizer: Optional[MapTransform] = ResizeWithPadOrCropd(
#             keys=CustomKeys.IMAGE, spatial_size=(512, 512, 256)
#         ),
#         cropper: Optional[MapTransform] = RandSpatialCropD(
#             keys=CustomKeys.IMAGE, roi_size=(96, 96, 96), random_size=False
#         ),
#         caster: Optional[MapTransform] = CastToTyped(
#             keys=[CustomKeys.IMAGE, CustomKeys.LABEL], dtype=[np.float32, np.int64]
#         ),
#         to_tensor: Optional[MapTransform] = ToTensord(
#             keys=[CustomKeys.IMAGE, CustomKeys.LABEL]
#         ),
#         dataset_type: Dataset = CacheDataset,
#         dataset_kwargs: dict = {'cache_rate': 1.0},
#         additional_transforms: Optional[Sequence] = None,
#         verbose: bool = False,
#     ):
#         assert loader is not None, "Loader cannot be None"

#         super().__init__(
#             files_list,
#             loader,
#             channeler,
#             orienter,
#             spacer,
#             rescaler,
#             resizer,
#             cropper,
#             caster,
#             to_tensor,
#             True,
#             dataset_type,
#             dataset_kwargs,
#             additional_transforms,
#             verbose,
#         )
