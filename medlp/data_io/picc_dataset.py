import os, sys, time, torch, random, tqdm
import numpy as np
from utils_cw import Print, load_h5
import nibabel as nib

#from dataio import load_picc_data_once
from scipy.ndimage.morphology import binary_dilation
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import monai
from monai.config import IndexSelection, KeysCollection
from monai.data import CacheDataset, Dataset
from monai.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple, InterpolateMode
from monai.transforms.utils import generate_pos_neg_label_crop_centers
from monai.transforms import *

from medlp.models.rcnn.structures.bounding_box import BoxList

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


def PICC_seg_dataset(files_list, phase, spacing=[], in_channels=1, image_size=None,
                     crop_size=None, preload=1.0, augment_ratio=0.4, downsample=1, verbose=False):

    data_reader = LoadHdf5d(keys=["image","label","coord"], h5_keys=["image","roi","coord"], 
                            affine_keys=["affine","affine",None], dtype=[np.float32, np.int64, np.float32])

    if in_channels > 1:
        repeater = RepeatChanneld(keys="image", repeats=in_channels)
    else:
        repeater = Lambdad(keys="image", func=lambda x : x)

    if spacing is None or spacing == [] or np.any(np.less_equal(spacing,0)):
        Print('No respacing!', color='g')
        spacer = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        spacer = Spacingd(keys=["image","label"], pixdim=spacing)

    if image_size is None or image_size == [] or np.any(np.less_equal(image_size,0)):
        Print('No resizing!', color='g')
        resizer = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        resizer = ResizeWithPadOrCropd(keys=["image","label"], spatial_size=image_size)

    if crop_size is None or crop_size == [] or np.any(np.less_equal(crop_size,0)):
        Print('No cropping!', color='g')
        cropper = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        cropper = RandCropByPosNegLabeld(keys=["image","label"], label_key='label', pos=2, neg=1, spatial_size=crop_size)

    if phase == 'train':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            spacer,
            resizer,
            cropper,
            #RandScaleIntensityd(keys="image",factors=(-0.01,0.01), prob=augment_ratio),
            RandAdjustContrastd(keys="image", prob=augment_ratio, gamma=(0.7,1.4)),
            #Rand2DElasticd(keys=["image","label"], prob=augment_ratio, spacing=(300, 300), magnitude_range=(10, 20), padding_mode="border"),
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
            cropper,
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
    dataset_ = CacheDataset(files_list, transform=transforms, cache_rate=preload)
    return dataset_


def RIB_seg_dataset(files_list, phase, in_channels=1, preload=1.0, image_size=None, 
                    crop_size=None, augment_ratio=0.4, downsample=1, verbose=False):

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
            ScaleIntensityd(keys="image"),
            ThresholdIntensityd(keys="label", threshold=1, above=False, cval=1),
            Zoomd(keys=["image", "label"], zoom=1/downsample, mode=[InterpolateMode.AREA,InterpolateMode.NEAREST], keep_size=False),
            #RandScaleIntensityd(keys="image",factors=(-0.01,0.01), prob=augment_ratio),
            RandAdjustContrastd(keys="image", prob=augment_ratio, gamma=(0.7,2.0)),
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
            ScaleIntensityd(keys=["image"]),
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
            ScaleIntensityd(keys=["image"]),
            Zoomd(keys=["image"], zoom=1/downsample, mode=[InterpolateMode.AREA], keep_size=False),
            DivisiblePadd(keys=["image"], k=32),
            CastToTyped(keys=["image"], dtype=[np.float32]),
            ToTensord(keys=["image"])
        ])
    dataset_ = CacheDataset(input_data, transform=transforms, cache_rate=preload)
    return dataset_


class DetDateSet(object):
    def __init__(self, file_list, bbox_radius=(20,20), transforms=None):
        self.file_list = file_list
        self.transforms = transforms
        self.bbox_radius = bbox_radius

    def __getitem__(self, idx):
        # load the image as a PIL Image
        image, coord = load_h5(self.file_list[idx], keywords=['image','coord'])

        # load the bounding boxes as a list of list of boxes
        # in this case, for illustrative purposes, we use x1, y1, x2, y2 order.
        boxes = [[coord[0]-self.bbox_radius[0], coord[1]-self.bbox_radius[1], 
                  coord[0]+self.bbox_radius[0], coord[1]+self.bbox_radius[1]]]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        # and labels
        labels = torch.tensor([1])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, image.shape, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        if self.transforms:
            image, boxlist = self.transforms(image, boxlist)

        # return the image, the boxlist and the idx in your dataset
        return image, boxlist, idx

    def get_img_info(self, idx):
        return None


def PICC_det_dataset(files_list, phase, in_channels=1, preload=1.0, image_size=None, 
                     crop_size=None, augment_ratio=0.4, downsample=1, verbose=False):
    pass