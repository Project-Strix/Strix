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

from strix.data_io import (
    SEGMENTATION_DATASETS,
    CLASSIFICATION_DATASETS,
    SELFLEARNING_DATASETS,
    StrixDataset,
)
from strix.utilities.enum import Frameworks

@SEGMENTATION_DATASETS.register("2D", "SyntheticData", None)
@SEGMENTATION_DATASETS.register("3D", "SyntheticData", None)
@SELFLEARNING_DATASETS.register("2D", "SyntheticData", None)
@SELFLEARNING_DATASETS.register("3D", "SyntheticData", None)
def synthetic_dataset(filelist, phase, opts):
    if opts['output_nc'] in [1, 2]:
        seg_cls = 1
    elif opts['output_nc'] > 2:
        seg_cls = opts['output_nc'] - 1
    else:
        raise ValueError(f"Got unexpected output_nc: {opts['output_nc']}")

    dim = opts["tensor_dim"]

    if opts["framework"] == Frameworks.SELFLEARNING.value:
        seg_cls = 0  # to generate two images not label

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

    return StrixDataset(
        filelist=filelist,
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
def random_dataset_cls_nc1(filelist, phase, opts):
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

    return StrixDataset(
        filelist,
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
