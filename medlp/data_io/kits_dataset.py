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

def get_kits_dataset(files_list, phase, spacing=[], in_channels=1, image_size=None,
                     crop_size=None, preload=1.0, augment_ratio=0.4, downsample=1, verbose=False):
    all_data = all_roi = files_list
    data_reader = LoadHdf5d(keys=["image","label"], h5_keys=["data","label"], dtype=[np.float32, np.int64])
    input_data = all_data

    if spacing:
        spacer = Spacingd(keys=["image","label"], pixdim=spacing)
    else:
        spacer = Lambdad(keys=["image", "label"], func=lambda x : x)

    if crop_size is None or np.any(np.less_equal(crop_size,0)):
        Print('No cropping!', color='g')
        cropper = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        cropper = =(keys=["image", "label"], roi_size=crop_size, random_size=False)

    if phase == 'train':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            RandAdjustContrastd(keys=["image","label"], prob=augment_ratio, gamma=(0.9,1.1)),
            cropper,
            RandGaussianNoised(keys="image", prob=augment_ratio, std=0.2),
            RandRotated(keys=["image","label"], range_x=10, range_y=10, range_z=5, prob=augment_ratio),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[0]),
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    elif phase == 'valid':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            cropper,
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    elif phase == 'test':
        pass


       
