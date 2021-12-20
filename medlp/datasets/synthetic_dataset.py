import os
import math
import numpy as np
from utils_cw import Print, check_dir

from monai_ex.data import Dataset
from monai_ex.transforms import LoadTestDataD, EnsureChannelFirstD, NormalizeIntensityD, CastToTypeD, ToTensorD

from medlp.data_io import (
    SEGMENTATION_DATASETS,
    CLASSIFICATION_DATASETS,
    BasicSegmentationDataset,
    BasicClassificationDataset,
)

@SEGMENTATION_DATASETS.register('3D', 'synthetic_nc1', "dummy.json")
def synthetic_dataset_3d(files_list, phase, opts):
    opts["out_cls"] = 1
    return synthetic_3d_dataset_generator(files_list, phase, opts)


@SEGMENTATION_DATASETS.register('3D', 'synthetic_nc2', "dummy.json")
def synthetic_dataset_3d(files_list, phase, opts):
    opts["out_cls"] = 2
    return synthetic_3d_dataset_generator(files_list, phase, opts)


def synthetic_3d_dataset_generator(files_list, phase, opts):
    out_cls = opts["out_cls"]
    return BasicSegmentationDataset(
        files_list,
        loader=LoadTestDataD(keys=['image', 'label'], width=64, height=64, depth=64, num_seg_classes=out_cls),
        channeler=EnsureChannelFirstD(keys=['image', 'label']),
        orienter=None,
        spacer=None,
        rescaler=NormalizeIntensityD(keys=['image']),
        resizer=None,
        cropper=None,
        caster=CastToTypeD(keys=['image', 'label'], dtype=[np.float32, np.int64]),
        to_tensor=ToTensorD(keys=['image', 'label']),
        is_supervised=True,
        dataset_type=Dataset,
        dataset_kwargs={},
        additional_transforms=None,
        check_data=False,
        verbose=True,
    ) 


@SEGMENTATION_DATASETS.register('2D', 'synthetic_nc2', "dummy.json")
def synthetic_dataset_2d(files_list, phase, opts):
    opts["out_cls"] = 2
    return synthetic_2d_dataset_generator(files_list, phase, opts)


@SEGMENTATION_DATASETS.register('2D', 'synthetic_nc1', "dummy.json")
def synthetic_dataset_3d(files_list, phase, opts):
    opts["out_cls"] = 1
    return synthetic_2d_dataset_generator(files_list, phase, opts)


def synthetic_2d_dataset_generator(files_list, phase, opts):
    out_cls = opts["out_cls"]
    return BasicSegmentationDataset(
        files_list,
        loader=LoadTestDataD(keys=['image', 'label'], width=64, height=64, num_seg_classes=out_cls),
        channeler=EnsureChannelFirstD(keys=['image', 'label']),
        orienter=None,
        spacer=None,
        rescaler=NormalizeIntensityD(keys=['image']),
        resizer=None,
        cropper=None,
        caster=CastToTypeD(keys=['image', 'label'], dtype=[np.float32, np.int64]),
        to_tensor=ToTensorD(keys=['image', 'label']),
        is_supervised=True,
        dataset_type=Dataset,
        dataset_kwargs={},
        additional_transforms=None,
        check_data=False,
        verbose=True,
    )


@CLASSIFICATION_DATASETS.register('3D', 'synthetic', "dummy.json")
def synthetic_dataset_cls_3d(files_list, phase, opts):
    pass


@CLASSIFICATION_DATASETS.register('2D', 'synthetic', "dummy.json")
def synthetic_dataset_cls_2d(files_list, phase, opts):
    pass
