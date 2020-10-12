import os, sys, time, torch, random, tqdm
import numpy as np
#from torch.utils.data import Dataset
from utils_cw import Print, load_h5
import nibabel as nib

# dataio_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append( os.path.join(os.path.dirname(dataio_dir), 'utils') )
# sys.path.append( dataio_dir )
#from dataio import load_picc_data_once
from scipy.ndimage.morphology import binary_dilation
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import monai
from monai.config import IndexSelection, KeysCollection
from monai.data import CacheDataset, Dataset
from monai.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple, InterpolateMode
from monai.transforms.utils import generate_pos_neg_label_crop_centers
from monai.transforms import (
    LoadHdf5d,
    LoadNumpyd,
    LoadNiftid,
    AddChanneld,
    AsChannelFirstd,
    SqueezeDimd,
    RepeatChanneld,
    Compose,
    Zoomd,
    Spacingd,
    SpatialCrop,
    ResizeWithPadOrCropd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandRotated,
    RandFlipd,
    Flipd,
    RandRotate90d,
    RandScaleIntensityd,
    Lambdad,
    Lambda,
    ToTensord,
    MapTransform, 
    Randomizable,
    CastToTyped,
    ThresholdIntensityd,
    NormalizeIntensityd,
    RandGaussianNoised,
    DivisiblePadd,
)


def load_picc_h5_data_once(file_list, h5_keys=['image', 'roi', 'coord'], transpose=None):
    #Pre-load all training data once.
    data = { i:[] for i in h5_keys }
    Print('\nPreload all {} training data'.format(len(file_list)), color='g')
    for fname in tqdm.tqdm(file_list):
        try:
            data_ = load_h5(fname, keywords=h5_keys, transpose=transpose)
            # if ds>1:
            #     data = data[::ds,::ds]
            #     roi  = roi[::ds,::ds]
            #     coord = coord[0]/ds, coord[1]/ds
        except Exception as e:
            Print('Data not exist!', fname, color='r')
            print(e)
            continue

        for i, key in enumerate(h5_keys):
           data[key].append(data_[i])
    return data.values()


def get_PICC_dataset(files_list, phase, spacing=[], in_channels=1, image_size=None,
                     crop_size=None, preload=True, augment_ratio=0.4, downsample=1, verbose=False):

    cache_ratio = 1.0 if preload else 0.0
    all_data = all_roi = all_coord = files_list
    data_reader = LoadHdf5d(keys=["image","label","coord"], h5_keys=["image","roi","coord"], 
                            affine_keys=["affine","affine",None], dtype=[np.float32, np.int64, np.float32])
    input_data = all_data

    if spacing:
        spacer = Spacingd(keys=["image","label"], pixdim=spacing)
    else:
        spacer = Lambdad(keys=["image", "label"], func=lambda x : x)

    if in_channels > 1:
        repeater = RepeatChanneld(keys="image", repeats=in_channels)
    else:
        repeater = Lambdad(keys="image", func=lambda x : x)

    if image_size is None or np.any(np.less_equal(image_size,0)):
        Print('No resizing!', color='g')
        resizer = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        resizer = ResizeWithPadOrCropd(keys=["image","label"], spatial_size=image_size)

    def debug(x):
        print("image type:", type(x), x.dtype, "image shape:", x.shape)
        return x
        nib.save(nib.Nifti1Image(x, np.eye(4)), f'./{str(time.time())}.nii.gz')
        return x
    debugger = Lambdad(keys=["image", "label"], func=debug)

    if phase == 'train':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            spacer,
            resizer,
            RandScaleIntensityd(keys="image",factors=(-0.01,0.01), prob=augment_ratio),
            RandRotated(keys=["image","label"], range_x=10, range_y=10, prob=augment_ratio),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[0]),
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            repeater,
            ToTensord(keys=["image", "label"]),
        ])
    elif phase == 'valid':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            spacer,
            resizer,
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    elif phase == 'test':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image"]),
            Spacingd(keys=["image"], pixdim=spacing),
            DivisiblePadd(keys=["image"], k=16),
            CastToTyped(keys=["image"], dtype=[np.float32]),
            ToTensord(keys=["image"])
        ])
    dataset_ = CacheDataset(input_data, transform=transforms, cache_rate=cache_ratio)
    return dataset_


def get_RIB_dataset(files_list, phase, in_channels=1, preload=True, image_size=None, 
                    crop_size=None, augment_ratio=0.4, downsample=1, verbose=False):

    cache_ratio = 1.0 if preload else 0.0
    input_data = []
    for img, msk in files_list:
        input_data.append({"image":img, "label":msk})

    if image_size is None or np.any(np.less_equal(image_size,0)):
        Print('No image resize!', color='g')
        resizer = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        resizer = ResizeWithPadOrCropd(keys=["image","label"], spatial_size=image_size)

    if crop_size is None or np.any(np.less_equal(crop_size,0)):
        Print('No cropping!', color='g')
        cropper = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        cropper = RandSpatialCropd(keys=["image", "label"], roi_size=crop_size, random_size=False)

    def change_win_level(x):
        image = x['image'].copy()
        filename = x["image_meta_dict"]["filename_or_obj"].split('/')[-1]
        if filename.startswith('1.2.'):
            pix_max, pix_min = 3873, 1256
        elif filename.startswith('1.3.'):
            pix_max, pix_min = 2515, 421
        else:
            pix_max, pix_min = 255, 105
        image[image>pix_max] = pix_max
        image[image<pix_min] = pix_min
        x['image'] = image
        return x
    winlevel = Lambda(func=change_win_level)

    if phase == 'train':
        transforms = Compose([
            LoadNiftid(keys=["image","label"]),
            AddChanneld(keys=["image", "label"]),
            NormalizeIntensityd(keys="image"),
            ThresholdIntensityd(keys="label", threshold=1, above=False, cval=1),
            Zoomd(keys=["image", "label"], zoom=1/downsample, mode=[InterpolateMode.AREA,InterpolateMode.NEAREST], keep_size=False),
            RandScaleIntensityd(keys="image",factors=(-0.01,0.01), prob=augment_ratio),
            resizer,
            cropper,
            RandGaussianNoised(keys="image", prob=augment_ratio, std=0.2),
            RandRotated(keys=["image","label"], range_x=10, range_y=10, prob=augment_ratio),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[0]),
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    elif phase == 'valid':
        transforms = Compose([
            LoadNiftid(keys=["image","label"]),
            AddChanneld(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"]),
            ThresholdIntensityd(keys=["label"], threshold=1, above=False, cval=1),
            Zoomd(keys=["image", "label"], zoom=1/downsample, mode=[InterpolateMode.AREA,InterpolateMode.NEAREST], keep_size=False),
            resizer,
            cropper,
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    elif phase == 'test':
        transforms = Compose([
            LoadNiftid(keys=["image"]),
            #AddChanneld(keys=["image"]),
            winlevel,
            Flipd(keys=["image"], spatial_axis=0),
            AsChannelFirstd(keys=['image']),
            NormalizeIntensityd(keys=["image"]),
            Zoomd(keys=["image"], zoom=1/downsample, mode=[InterpolateMode.AREA], keep_size=False),
            DivisiblePadd(keys=["image"], k=32),
            CastToTyped(keys=["image"], dtype=[np.float32]),
            ToTensord(keys=["image"])
        ])
    dataset_ = CacheDataset(input_data, transform=transforms, cache_rate=cache_ratio)
    return dataset_

