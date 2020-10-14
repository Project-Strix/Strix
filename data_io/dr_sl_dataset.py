import os, sys, time, torch, random, tqdm
import numpy as np
import nibabel as nib
from utils_cw import Print, load_h5
from scipy.ndimage.morphology import binary_dilation
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import monai
from monai.config import IndexSelection, KeysCollection
from monai.data import CacheDataset, Dataset
from monai.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple, InterpolateMode
from monai.transforms.utils import generate_pos_neg_label_crop_centers
from monai.transforms import *


def get_ObjCXR_dataset(files_list, phase, in_channels=1, preload=1.0, image_size=None, 
                       crop_size=None, augment_ratio=0.5, downsample=1, verbose=False):
    input_data = []
    for img in files_list:
        input_data.append({"image":img, "label":img})

    if image_size is None or np.any(np.less_equal(image_size,0)):
        Print('No image resizing!', color='g')
        resizer = Lambda(func=lambda x : x)
    else:
        resizer = ResizeWithPadOrCropd(keys=["image","label"], spatial_size=image_size)

    if phase == 'train' or phase == 'valid':
        transforms = Compose([
            LoadPNGd(keys=["image","label"]),
            AddChanneld(keys=["image","label"]),
            ScaleIntensityd(keys=["image","label"]),
            FixedResized(keys=["image","label"], spatial_size=image_size[0], mode=[InterpolateMode.AREA,InterpolateMode.AREA]),
            resizer,
            RandAdjustContrastd(keys=["image","label"], prob=augment_ratio, gamma=(0.5, 2)),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[1]),
            Rand2DElasticd(keys="image", prob=augment_ratio, spacing=(300, 300), magnitude_range=(10, 20), padding_mode="border"),
            adaptor(RandLocalPixelShuffle(prob=augment_ratio, num_block_range=[50,200]), "image"),
            RandomSelect([
                adaptor(RandImageInpainting(prob=1, num_block_range=(3,5)), 'image'),
                adaptor(RandImageOutpainting(prob=1, num_block_range=(5,8)), 'image'),
            ], prob=0.7),
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.float32]),
            ToTensord(keys=["image","label"])
        ])
    elif phase == 'test':
        raise NotImplementedError
    
    dataset_ = CacheDataset(input_data, transform=transforms, cache_rate=preload)
    return dataset_