import numpy as np

from monai_ex.data import Dataset
from monai_ex.transforms import (
    GenerateSyntheticDataD,
    GenerateRandomDataD,
    EnsureChannelFirstD,
    NormalizeIntensityD,
    CastToTypeD,
    ToTensorD,
)

from medlp.data_io import (
    SEGMENTATION_DATASETS,
    CLASSIFICATION_DATASETS,
    BasicSegmentationDataset,
    BasicClassificationDataset,
)


@SEGMENTATION_DATASETS.register("2D", "SyntheticData", None)
@SEGMENTATION_DATASETS.register("3D", "SyntheticData", None)
def synthetic_dataset_3d(files_list, phase, opts):
    if opts['output_nc'] in [1, 2]:
        seg_cls = 1
    elif opts['output_nc'] > 2:
        seg_cls = opts['output_nc'] - 1
    else:
        raise ValueError(f"Got unexpected output_nc: {opts['output_nc']}")

    dim = opts["tensor_dim"]

    if dim == "2D":
        loader = GenerateSyntheticDataD(
            keys=["image", "label"],
            width=64, height=64,
            num_seg_classes=seg_cls
        )
    elif dim == "3D":
        loader = GenerateSyntheticDataD(
            keys=["image", "label"],
            width=64, height=64, depth=64,
            num_seg_classes=seg_cls,
        )

    return BasicSegmentationDataset(
        files_list,
        loader=loader,
        channeler=EnsureChannelFirstD(keys=["image", "label"]),
        orienter=None,
        spacer=None,
        rescaler=NormalizeIntensityD(keys=["image"]),
        resizer=None,
        cropper=None,
        caster=CastToTypeD(keys=["image", "label"], dtype=[np.float32, np.int64]),
        to_tensor=ToTensorD(keys=["image", "label"]),
        is_supervised=True,
        dataset_type=Dataset,
        dataset_kwargs={},
        additional_transforms=None,
        check_data=False,
        verbose=True,
    )


@CLASSIFICATION_DATASETS.register("2D", "RandomData", None)
@CLASSIFICATION_DATASETS.register("3D", "RandomData", None)
def random_dataset_cls_nc1(files_list, phase, opts):
    if opts['output_nc'] in [1, 2]:
        out_cls = 1
    elif opts['output_nc'] > 2:
        out_cls = opts['output_nc'] - 1
    else:
        raise ValueError(f"Got unexpected output_nc: {opts['output_nc']}")

    dim = opts["tensor_dim"]

    if dim == "2D":
        loader = GenerateRandomDataD(
            keys=["image", "label"], width=64, height=64, num_classes=out_cls
        )
    elif dim == "3D":
        loader = GenerateRandomDataD(
            keys=["image", "label"], width=64, height=64, depth=32, num_classes=out_cls
        )

    return BasicClassificationDataset(
        files_list,
        loader=loader,
        channeler=EnsureChannelFirstD(keys="image"),
        orienter=None,
        spacer=None,
        rescaler=NormalizeIntensityD(keys="image"),
        resizer=None,
        cropper=None,
        caster=CastToTypeD(keys="image", dtype=np.float32),
        to_tensor=ToTensorD(keys=["image", "label"]),
        is_supervised=True,
        dataset_type=Dataset,
        dataset_kwargs={},
        additional_transforms=None,
        check_data=False,
        verbose=True,
    )
