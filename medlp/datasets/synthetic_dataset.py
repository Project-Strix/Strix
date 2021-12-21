import numpy as np
from utils_cw import Print, check_dir

from monai_ex.data import Dataset
from monai_ex.transforms import GenerateSyntheticDataD, GenerateRandomDataD, EnsureChannelFirstD, NormalizeIntensityD, CastToTypeD, ToTensorD
from monai.transforms.transform import MapTransform

from medlp.data_io import (
    SEGMENTATION_DATASETS,
    CLASSIFICATION_DATASETS,
    BasicSegmentationDataset,
    BasicClassificationDataset,
)

@SEGMENTATION_DATASETS.register('2D', 'synthetic_nc1', "dummy.json")
@SEGMENTATION_DATASETS.register('3D', 'synthetic_nc1', "dummy.json")
def synthetic_dataset_3d(files_list, phase, opts):
    opts["out_cls"] = 1
    return synthetic_dataset_generator(files_list, phase, opts)


@SEGMENTATION_DATASETS.register('2D', 'synthetic_nc2', "dummy.json")
@SEGMENTATION_DATASETS.register('3D', 'synthetic_nc2', "dummy.json")
def synthetic_dataset_3d(files_list, phase, opts):
    opts["out_cls"] = 2
    return synthetic_dataset_generator(files_list, phase, opts)


def synthetic_dataset_generator(files_list, phase, opts):
    out_cls = opts["out_cls"]
    dim = opts['tensor_dim']

    if dim == '2D':
        loader = GenerateSyntheticDataD(keys=['image', 'label'], width=64, height=64, num_seg_classes=out_cls)
    elif dim == '3D':
        loader = GenerateSyntheticDataD(keys=['image', 'label'], width=64, height=64, depth=64, num_seg_classes=out_cls)

    return BasicSegmentationDataset(
        files_list,
        loader=loader,
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


@CLASSIFICATION_DATASETS.register('2D', 'random_nc1', "dummy.json")
@CLASSIFICATION_DATASETS.register('3D', 'random_nc1', "dummy.json")
def random_dataset_cls_nc1(files_list, phase, opts):
    opts["out_cls"] = 1
    return random_cls_dataset_generator(files_list, phase, opts)


@CLASSIFICATION_DATASETS.register('2D', 'random_nc2', "dummy.json")
@CLASSIFICATION_DATASETS.register('3D', 'random_nc2', "dummy.json")
def random_dataset_cls_nc2(files_list, phase, opts):
    opts["out_cls"] = 2
    return random_cls_dataset_generator(files_list, phase, opts)


def random_cls_dataset_generator(files_list, phase, opts):
    out_cls = opts["out_cls"]
    dim = opts['tensor_dim']

    if dim == '2D':
        loader = GenerateRandomDataD(
            keys=['image', 'label'], width=64, height=64, num_classes=out_cls
        )
    elif dim == '3D':
        loader = GenerateRandomDataD(
            keys=['image', 'label'], width=64, height=64, depth=32, num_classes=out_cls
        )

    return BasicClassificationDataset(
        files_list,
        loader=loader,
        channeler=EnsureChannelFirstD(keys='image'),
        orienter=None,
        spacer=None,
        rescaler=NormalizeIntensityD(keys=['image']),
        resizer=None,
        cropper=None,
        caster=CastToTypeD(keys='image', dtype=np.float32),
        to_tensor=ToTensorD(keys=['image', 'label']),
        is_supervised=True,
        dataset_type=Dataset,
        dataset_kwargs={},
        additional_transforms=None,
        check_data=False,
        verbose=True,
    )

