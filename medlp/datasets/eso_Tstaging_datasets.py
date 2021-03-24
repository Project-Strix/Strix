import os, sys, time, torch, random, tqdm, math
import numpy as np
from utils_cw import Print, load_h5, check_dir
import nibabel as nib

# from dataio import load_picc_data_once
from scipy.ndimage.morphology import binary_dilation
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from monai_ex.data import CacheDataset, Dataset
from monai_ex.utils import (
    Method,
    NumpyPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    InterpolateMode,
)
from monai_ex.transforms import *

from medlp.models.rcnn.structures.bounding_box import BoxList
from medlp.data_io import (
    CLASSIFICATION_DATASETS,
    SEGMENTATION_DATASETS,
)
from medlp.data_io import BasicClassificationDataset
from medlp.utilities.utils import is_avaible_size
from medlp.utilities.transforms import *


@CLASSIFICATION_DATASETS.register("2D", "eso_vibe",
    "/homes/yliu/Data/clwang_data/survival_new/feat/tp_exp/json_file/sg_vibe_1mm_train_Tstaging.json")
def Survival_nii_cls_dataset(files_list, phase, opts):
    spacing = opts.get("spacing", (1, 1, 3))
    # in_channels = opts.get("input_nc", 3)
    preload = opts.get("preload", 0)
    augment_ratio = opts.get("augment_ratio", 0.4)
    orientation = opts.get("orientation", "RPI")
    cache_dir = check_dir(opts.get("experiment_path"), "caches")

    cropper = [
        CenterMask2DSliceCropD(
            keys=["image", "mask"],
            mask_key="mask",
            roi_size=(48, 48),
            z_axis=2,
            crop_mode="parallel",
            center_mode="maximum",
        ),
        # GetMaxSlices3direcD(keys=['image','mask'], mask_key='mask'),
        # RandCrop2dByPosNegLabelD(keys=['image','mask'], label_key='mask', spatial_size=(48,48), z_axis=2, crop_mode='parallel', pos=1, neg=0),
        # Mergetargetevent(key1="survival_time", key2="event"),
        # LabelMorphologyD(keys='mask',mode='dilation',radius=1,binary=False),
        # MaskIntensityExD(keys="image", mask_key="mask")
    ]
    if phase == "train":
        additional_transforms = [
            RandFlipd(keys=["image", "mask"], prob=augment_ratio, spatial_axis=0),
            RandRotateD(
                keys=["image", "mask"],
                range_x=math.pi / 18,
                range_y=math.pi / 18,
                prob=augment_ratio,
            ),
            RandGaussianNoised(keys="image", prob=augment_ratio, std=0.01),
        ]
    elif phase == "valid":
        additional_transforms = []
    else:
        additional_transforms = []

    dataset = BasicClassificationDataset(
        files_list,
        loader=LoadNiftiD(keys=["image", "mask"]),
        channeler=AddChanneld(keys=["image", "mask"]),
        orienter=Orientationd(keys=["image", "mask"], axcodes=orientation),
        spacer=SpacingD(keys=["image", "mask"], pixdim=spacing),
        rescaler=[
            ScaleIntensityRangePercentilesD(
                keys="image", lower=0.0, upper=99.5, b_min=0, b_max=1
            ),
            LambdaD(keys='label', func=lambda x: x > 2),
        ],
        resizer=None,
        cropper=cropper,
        additional_transforms=additional_transforms,
        caster=CastToTyped(keys=["image", "mask"], dtype=[np.float32, np.int16]),
        to_tensor=ToTensord(keys=["image", "mask"]),
        is_supervised=True,
        dataset_type=CacheDataset,
        dataset_kwargs={'cache_rate': preload},
    ).get_dataset()

    return dataset