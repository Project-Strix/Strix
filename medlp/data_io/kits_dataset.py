import os, sys, time, torch, random, tqdm
import numpy as np
from utils_cw import Print, load_h5
import nibabel as nib

from scipy.ndimage.morphology import binary_dilation
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import monai
from monai.config import IndexSelection, KeysCollection
from monai.data import CacheDataset, Dataset
from monai.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple, InterpolateMode
from monai.transforms.utils import generate_pos_neg_label_crop_centers
from monai.transforms import *

def get_kits_dataset(files_list, phase, spacing=[], winlevel=[-80,304], in_channels=1, image_size=None,
                     crop_size=None, preload=1.0, augment_ratio=0.4, downsample=1, verbose=False):
    #data_reader = LoadHdf5d(keys=["image","label"], h5_keys=["data","label"], dtype=[np.float32, np.int64])
    data_reader = LoadNiftid(keys=["image","label"], dtype=np.float32)

    if spacing:
        spacer = Spacingd(keys=["image","label"], pixdim=spacing)
    else:
        spacer = Lambdad(keys=["image","label"], func=lambda x : x)

    if crop_size is None or np.any(np.less_equal(crop_size,0)):
        Print('No cropping!', color='g')
        cropper = Lambdad(keys=["image","label"], func=lambda x : x)
    else:
        cropper = RandCropByPosNegLabeld(keys=["image","label"], label_key='label', spatial_size=crop_size)

    normalizer = ScaleIntensityRanged(keys="image", a_min=winlevel[0], a_max=winlevel[1], b_min=0, b_max=1, clip=True)

    if phase == 'train':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            normalizer,
            spacer,
            cropper,
            Rand3DElasticd(keys="image", prob=augment_ratio, sigma_range=(5,10), 
                           magnitude_range=(50,150), padding_mode='zeros', device=torch.device('cuda')),
            RandGaussianNoised(keys="image", prob=augment_ratio, std=0.02),
            RandRotated(keys=["image","label"], range_x=5, range_y=5, range_z=10, prob=augment_ratio, padding_mode='zeros'),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=0),
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    elif phase == 'valid':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            normalizer,
            spacer,
            cropper,
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    elif phase == 'test':
        pass

    dataset_ = CacheDataset(files_list, transform=transforms, cache_rate=preload)
    return dataset_
       
