import os
import numpy as np
from typing import Optional, Sequence, Union

# from monai_ex.engines.utils import CustomKeys, get_keys_list
from monai_ex.data import Dataset
from monai_ex.transforms import *
from monai_ex.utils import ensure_list
from medlp.data_io.base_dataset.utils import get_input_data


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


# class SupervisedSegmentationDataset(BasicSegmentationDataset):
#     def __init__(
#         self,
#         files_list,
#         loader: MapTransform = LoadNiftid(
#             keys=[CustomKeys.IMAGE, CustomKeys.LABEL], dtype=np.float32
#         ),
#         channeler: Optional[MapTransform] = AddChanneld(
#             keys=[CustomKeys.IMAGE, CustomKeys.LABEL]
#         ),
#         orienter: Optional[MapTransform] = Orientationd(
#             keys=[CustomKeys.IMAGE, CustomKeys.LABEL], axcodes="LPI"
#         ),
#         spacer: Optional[MapTransform] = Spacingd(
#             keys=[CustomKeys.IMAGE, CustomKeys.LABEL], pixdim=(0.1, 0.1)
#         ),
#         rescaler: Optional[MapTransform] = ScaleIntensityRanged(
#             keys=[CustomKeys.IMAGE], a_min=0, a_max=255, b_min=0, b_max=1, clip=True
#         ),
#         resizer: Optional[MapTransform] = ResizeWithPadOrCropd(
#             keys=[CustomKeys.IMAGE, CustomKeys.LABEL], spatial_size=(768, 768)
#         ),
#         cropper: Optional[MapTransform] = RandSpatialCropd(
#             keys=[CustomKeys.IMAGE, CustomKeys.LABEL], roi_size=(384, 384), random_size=False
#         ),
#         caster: Optional[MapTransform] = CastToTyped(
#             keys=[CustomKeys.IMAGE, CustomKeys.LABEL], dtype=[np.float32, np.int64]
#         ),
#         to_tensor: Optional[MapTransform] = ToTensord(keys=[CustomKeys.IMAGE, CustomKeys.LABEL]),
#         dataset_type: Dataset = CacheDataset,
#         dataset_kwargs: dict = {'cache_rate': 1},
#         additional_transforms: Optional[Sequence] = None,
#         verbose: bool = False,
#     ):
#         assert loader is not None, "Loader cannot be None"

#         super().__init__(
#             files_list=files_list,
#             loader=loader,
#             channeler=channeler,
#             orienter=orienter,
#             spacer=spacer,
#             rescaler=rescaler,
#             resizer=resizer,
#             cropper=cropper,
#             caster=caster,
#             to_tensor=to_tensor,
#             is_supervised=True,
#             dataset_type=dataset_type,
#             dataset_kwargs=dataset_kwargs,
#             additional_transforms=additional_transforms,
#             verbose=verbose,
#         )


# class UnsupervisedSegmentationDataset(BasicSegmentationDataset):
#     def __init__(
#         self,
#         files_list,
#         loader: MapTransform = LoadNiftid(
#             keys=[CustomKeys.IMAGE], dtype=np.float32
#         ),
#         channeler: Optional[MapTransform] = AddChanneld(
#             keys=[CustomKeys.IMAGE]
#         ),
#         orienter: Optional[MapTransform] = Orientationd(
#             keys=[CustomKeys.IMAGE], axcodes="LPI"
#         ),
#         spacer: Optional[MapTransform] = Spacingd(
#             keys=[CustomKeys.IMAGE], pixdim=(0.1, 0.1)
#         ),
#         rescaler: Optional[MapTransform] = ScaleIntensityRanged(
#             keys=[CustomKeys.IMAGE], a_min=0, a_max=255, b_min=0, b_max=1, clip=True
#         ),
#         resizer: Optional[MapTransform] = ResizeWithPadOrCropd(
#             keys=[CustomKeys.IMAGE], spatial_size=(768, 768)
#         ),
#         cropper: Optional[MapTransform] = RandSpatialCropd(
#             keys=[CustomKeys.IMAGE], roi_size=(384, 384), random_size=False
#         ),
#         caster: Optional[MapTransform] = CastToTyped(
#             keys=[CustomKeys.IMAGE], dtype=np.float32
#         ),
#         to_tensor: Optional[MapTransform] = ToTensord(
#             keys=[CustomKeys.IMAGE]
#         ),
#         dataset_type: Dataset = CacheDataset,
#         dataset_kwargs: dict = {'cache_rate': 1.0},
#         additional_transforms: Optional[Sequence] = None,
#         verbose: bool = False,
#     ):
#         assert loader is not None, "Loader cannot be None"

#         super().__init__(
#             files_list=files_list,
#             loader=loader,
#             channeler=channeler,
#             orienter=orienter,
#             spacer=spacer,
#             rescaler=rescaler,
#             resizer=resizer,
#             cropper=cropper,
#             caster=caster,
#             to_tensor=to_tensor,
#             is_supervised=False,
#             dataset_type=dataset_type,
#             dataset_kwargs=dataset_kwargs,
#             additional_transforms=additional_transforms,
#             verbose=verbose,
#         )
