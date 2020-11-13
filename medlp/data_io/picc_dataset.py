import os, sys, time, torch, random, tqdm, math
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
from medlp.data_io.segmentation_dataset import SegmentationDataset2D
from medlp.utilities.utils import is_avaible_size

def PICC_dcm_seg_dataset(
    files_list, 
    phase, 
    spacing=(0.3,0.3), 
    in_channels=1, 
    image_size=(1024,1024),
    crop_size=None, 
    preload=1.0, 
    augment_ratio=0.4
):
    assert in_channels == 1, 'Currently only support single channel input'
    
    if phase == 'train':
        additional_transforms = [
            RandRotated(keys=["image","label"], range_x=math.pi/20, range_y=math.pi/20, prob=augment_ratio, padding_mode='zeros'),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[1])
        ]
    elif phase == 'valid':
        additional_transforms = []

    ignore_dcm_keys = ['0040|0244','0040|0245','0040|0253','0040|0254','0032|1060']
    dataset = SegmentationDataset2D(
        files_list,
        loader=[LoadImageD(keys='image', drop_meta_keys=ignore_dcm_keys), LoadNiftiD(keys='label')],
        channeler=TransposeD(keys='label'), #to match the axes of itk and numpy
        orienter=None,
        spacer=SpacingD(keys=["image","label"], pixdim=spacing),
        rescaler=ScaleIntensityViaDicomD(keys="image", win_center_key='0028|1050', win_width_key='0028|1051', clip=True),
        resizer=ResizeD(keys=["image","label"], spatial_size=image_size) if is_avaible_size(image_size) else None,#ResizeWithPadOrCropd(keys=["image","label"], spatial_size=image_size) 
        cropper=RandSpatialCropd(keys=["image", "label"], roi_size=crop_size, random_size=False) if is_avaible_size(crop_size) else None,
        additional_transforms=additional_transforms,    
        preload=preload
    ).get_dataset()

    return dataset

def PICC_nii_seg_dataset(
    files_list, 
    phase, 
    spacing=(0.3,0.3), 
    winlevel=(421,2515), 
    in_channels=1, 
    image_size=(1024,1024),
    crop_size=None, 
    preload=1.0, 
    augment_ratio=0.4
):
    assert in_channels == 1, 'Currently only support single channel input'
    
    if phase == 'train':
        additional_transforms = [
            RandRotated(keys=["image","label"], range_x=math.pi/20, range_y=math.pi/20, prob=augment_ratio, padding_mode='zeros'),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[1])
        ]
    elif phase == 'valid':
        additional_transforms = []

    dataset = SegmentationDataset2D(
        files_list,
        loader=LoadNiftiD(keys=['image','label']),
        channeler=AsChannelFirstD(keys=['image','label']),
        spacer=SpacingD(keys=["image","label"], pixdim=spacing),
        rescaler=ScaleIntensityRangeD(keys=["image"], a_min=winlevel[0], a_max=winlevel[1], b_min=0, b_max=1, clip=True),
        resizer=ResizeWithPadOrCropD(keys=["image","label"], spatial_size=image_size) if is_avaible_size(image_size) else None,
        cropper=RandSpatialCropd(keys=["image", "label"], roi_size=crop_size, random_size=False) if is_avaible_size(crop_size) else None,
        additional_transforms=additional_transforms,    
        preload=preload
    ).get_dataset()

    return dataset


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
            RandRotated(keys=["image","label"], range_x=math.pi/20, range_y=math.pi/20, prob=augment_ratio),
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
            RandAdjustContrastd(keys="image", prob=augment_ratio, gamma=(0.8,1.8)),
            resizer,
            cropper,
            RandGaussianNoised(keys="image", prob=augment_ratio, std=0.2),
            RandRotated(keys=["image","label"], range_x=math.pi/18, range_y=math.pi/18, prob=augment_ratio),
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


class DetDateSet(Dataset):
    def __init__(self, file_list, bbox_radius=(20,20), transforms=None):
        self.file_list = file_list
        self.transforms = transforms
        self.bbox_radius = bbox_radius

    def __len__(self) -> int:
        return len(self.file_list)

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