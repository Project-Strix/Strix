import numpy as np
from utils_cw import Print
from scipy.ndimage.morphology import binary_dilation
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

from monai_ex.data import CacheDataset, Dataset
from monai_ex.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple, InterpolateMode
from monai_ex.transforms import *

from medlp.data_io import SELFLEARNING_DATASETS

@SELFLEARNING_DATASETS.register('2D','ObjCXR',"/homes/clwang/Data/object-CXR/train_data_list.json")
def get_ObjCXR_dataset(files_list, phase, opts):
    in_channels=opts.get('input_nc', 1)
    preload=opts.get('preload', 1.0)
    image_size=opts.get('image_size', None)
    crop_size=opts.get('crop_size', None)
    augment_ratio=opts.get('augment_ratio', 0.5)

    input_data = []
    for img in files_list:
        input_data.append({"image":img, "label":img})

    if image_size is None or np.any(np.less_equal(image_size,0)):
        Print('No image resizing!', color='g')
        resizer = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        resizer = ResizeWithPadOrCropd(keys=["image","label"], spatial_size=image_size)

    if crop_size is None or np.any(np.less_equal(crop_size,0)):
        Print('No image cropping!', color='g')
        cropper = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        cropper = RandSpatialCropd(keys=["image", "label"], roi_size=crop_size, random_size=False)

    if phase == 'train' or phase == 'valid':
        transforms = Compose([
            LoadPNGd(keys=["image","label"], grayscale=True),
            AddChanneld(keys=["image","label"]),
            ScaleIntensityd(keys=["image","label"]),
            FixedResized(keys=["image","label"], spatial_size=image_size[0], mode=[InterpolateMode.AREA,InterpolateMode.AREA]),
            resizer,
            cropper,
            #RandAdjustContrastd(keys=["image","label"], prob=augment_ratio, gamma=(0.5, 1.5)),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[1]),
            adaptor(RandNonlinear(prob=augment_ratio), "image"),
            adaptor(RandLocalPixelShuffle(prob=augment_ratio, num_block_range=(9000,10000)), "image"),
            RandomSelect([
                adaptor(RandImageInpainting(prob=1, num_block_range=(3,5)), 'image'),
                adaptor(RandImageOutpainting(prob=1, num_block_range=(6,9)), 'image'),
            ], prob=augment_ratio),
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.float32]),
            ToTensord(keys=["image","label"])
        ])
    elif phase == 'test':
        raise NotImplementedError
    
    dataset_ = CacheDataset(input_data, transform=transforms, cache_rate=preload)
    return dataset_

@SELFLEARNING_DATASETS.register('2D','NIHXray',"/homes/clwang/Data/NIH-CXR_TRAIN_VAL_LIST.json")
def get_NIHXray_dataset(files_list, phase, opts):
    in_channels=opts.get('input_nc', 1)
    preload=opts.get('preload', 1.0)
    image_size=opts.get('image_size', None)
    crop_size=opts.get('crop_size', None)
    augment_ratio=opts.get('augment_ratio', 0.5)

    input_data = []
    for img in files_list:
        input_data.append({"image":img, "label":img})    
    
    if image_size is None or np.any(np.less_equal(image_size,0)):
        Print('No image resizing!', color='g')
        resizer = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        resizer = ResizeWithPadOrCropd(keys=["image","label"], spatial_size=image_size)

    if crop_size is None or np.any(np.less_equal(crop_size,0)):
        Print('No image cropping!', color='g')
        cropper = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        cropper = RandSpatialCropd(keys=["image", "label"], roi_size=crop_size, random_size=False)

    if phase == 'train' or phase == 'valid':
        transforms = Compose([
            LoadPNGd(keys=["image","label"], grayscale=True),
            AddChanneld(keys=["image","label"]),
            ScaleIntensityd(keys=["image","label"]),
            resizer,
            cropper,
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[1]),
            adaptor(RandNonlinear(prob=augment_ratio), "image"),
            adaptor(RandLocalPixelShuffle(prob=augment_ratio, num_block_range=[9000,10000]), "image"),
            RandomSelect([
                adaptor(RandImageInpainting(prob=1, num_block_range=(3,5)), 'image'),
                adaptor(RandImageOutpainting(prob=1, num_block_range=(6,9)), 'image'),
            ], prob=augment_ratio),
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.float32]),
            ToTensord(keys=["image","label"])
        ])
    elif phase == 'test':
        raise NotImplementedError
    
    dataset_ = CacheDataset(input_data, transform=transforms, cache_rate=preload)
    return dataset_

