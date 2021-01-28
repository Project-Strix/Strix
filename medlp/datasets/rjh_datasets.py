import os, sys, time, torch, random, tqdm, math
import numpy as np
from utils_cw import Print, load_h5, check_dir
import nibabel as nib

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from medlp.data_io import CLASSIFICATION_DATASETS, SEGMENTATION_DATASETS
from medlp.data_io.base_dataset.segmentation_dataset import SupervisedSegmentationDataset3D, UnsupervisedSegmentationDataset3D
from medlp.data_io.base_dataset.classification_dataset import SupervisedClassificationDataset3D, BasicClassificationDataset
from medlp.utilities.utils import is_avaible_size
from medlp.utilities.transforms import (
    DataLabellingD,
    RandLabelToMaskD,
    SeparateCropSTSdataD,
    ExtractSTSlicesD
)

from monai_ex.data import CacheDataset, Dataset, PersistentDataset
from monai_ex.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple, InterpolateMode
from monai_ex.transforms import *


@SEGMENTATION_DATASETS.register('3D', 'rjh_tswi',
    "/homes/clwang/Data/RJH/RJ_data/preprocessed/labeled_data_list.json")
def get_rjh_tswi_seg_dataset(files_list, phase, opts):
    spacing=(0.666667,0.666667,1.34)
    in_channels=opts.get('input_nc', 1)
    crop_size=opts.get('crop_size', (96,96,64))
    preload=opts.get('preload', 0)
    augment_ratio=opts.get('augment_ratio',0.4)
    orientation='RAI'
    cache_dir=check_dir(os.path.dirname(opts.get('experiment_path')),'caches')

    assert in_channels == 1, 'Currently only support single channel input'

    cropper = RandCropByPosNegLabeld(keys=["image","label"], label_key='label', pos=1, neg=1, spatial_size=crop_size) if \
              is_avaible_size(crop_size) else None
    
    if phase == 'train':
        additional_transforms = [
            #RandFlipD(keys=["image","label"], prob=augment_ratio, spatial_axis=[2]),
            RandRotateD(keys=["image","label"], range_x=math.pi/40, range_y=math.pi/40, range_z=math.pi/40, prob=augment_ratio, padding_mode='zeros'),
            Rand3DElasticD(keys=["image","label"], prob=augment_ratio, sigma_range=(5,10),
                           magnitude_range=(50,150), mode=["bilinear","nearest"], padding_mode='zeros')
        ]
    elif phase == 'valid':
        additional_transforms = []
    elif phase == 'test':
        cropper = None
        additional_transforms = []
    elif phase == 'test_wo_label':
        return UnsupervisedSegmentationDataset3D(
                    files_list,
                    orienter=Orientationd(keys='image', axcodes=orientation),
                    spacer=SpacingD(keys="image", pixdim=spacing),
                    resizer=None,
                    rescaler=NormalizeIntensityD(keys='image'),
                    cropper=None,
                    additional_transforms=[],
                    preload=0,
                    cache_dir=cache_dir,
                ).get_dataset()
    else:
        raise ValueError

    dataset = SupervisedSegmentationDataset3D(
        files_list,
        orienter=Orientationd(keys=['image', 'label'], axcodes=orientation),
        spacer=SpacingD(keys=["image", "label"], pixdim=spacing, mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]),
        resizer=None,
        rescaler=[NormalizeIntensityD(keys='image'),]+[DataLabellingD(keys='label'),
                                                       LabelMorphologyD(keys='label',mode='dilation',radius=1,binary=False)],
        cropper=cropper,
        additional_transforms=additional_transforms,
        preload=preload,
        cache_dir=cache_dir,
    ).get_dataset()

    return dataset


@SEGMENTATION_DATASETS.register('3D', 'rjh_tswi_v2',
    "/homes/clwang/Data/RJH/RJ_data/tSWI_preprocessed/train_datalist_sn.json")
def get_rjh_tswi_roi45(files_list, phase, opts):
    spacing = (0.666667, 0.666667, 1.34)
    in_channels = opts.get('input_nc', 1)
    crop_size = opts.get('crop_size', (96, 96, 64))
    preload = opts.get('preload', 0)
    augment_ratio = opts.get('augment_ratio', 0.4)
    orientation = 'RAI'
    cache_dir = check_dir(os.path.dirname(opts.get('experiment_path')),'caches')

    assert in_channels == 1, 'Currently only support single channel input'

    cropper = []
    if is_avaible_size(crop_size):
        cropper += [RandCropByPosNegLabeld(keys=["image", "label"], label_key='label', pos=1, neg=1, spatial_size=crop_size)]

    if phase == 'train':
        additional_transforms = [
            RandAdjustContrastD(
                keys='image',
                prob=augment_ratio,
                gamma=(0.7, 1.3),
            ),
            RandRotateD(
                keys=["image", "label"],
                range_x=math.pi/40,
                range_y=math.pi/40,
                range_z=math.pi/40,
                prob=augment_ratio,
                mode=["bilinear", "nearest"],
                padding_mode='border'
            )
        ]
    elif phase == 'valid':
        additional_transforms = []
    elif phase == 'test':
        cropper = None
        additional_transforms = []
    elif phase == 'test_wo_label':
        return UnsupervisedSegmentationDataset3D(
                    files_list,
                    orienter=Orientationd(keys='image', axcodes=orientation),
                    spacer=SpacingD(keys="image", pixdim=spacing),
                    resizer=None,  #CropForegroundD(keys="image", source_key="image"),
                    rescaler=NormalizeIntensityD(keys='image'),
                    cropper=None,
                    additional_transforms=[],
                    preload=0,
                    cache_dir=cache_dir,
                ).get_dataset()
    else:
        raise ValueError
    
    dataset = SupervisedSegmentationDataset3D(
        files_list,
        orienter=Orientationd(keys=['image','label'], axcodes=orientation),
        spacer=SpacingD(keys=["image","label"], pixdim=spacing, mode=[GridSampleMode.BILINEAR,GridSampleMode.NEAREST]),
        resizer=None,  #CropForegroundD(keys=["image","label"], source_key="image"),
        rescaler=NormalizeIntensityD(keys='image'),
        cropper=cropper,
        additional_transforms=additional_transforms,
        preload=preload,
        cache_dir=cache_dir,
    ).get_dataset()

    return dataset


@SEGMENTATION_DATASETS.register('3D','rjh_swim', 
    "/homes/clwang/Data/RJH/RJ_data/SWIM_preprocessed/swim_train.json")
def get_rjh_swim_seg_dataset(files_list, phase, opts):
    spacing=(0.666667,0.666667,2)
    in_channels=opts.get('input_nc', 1)
    crop_size=opts.get('crop_size', (96,96,48))
    preload=opts.get('preload', 0)
    augment_ratio=opts.get('augment_ratio',0.4)
    orientation='RAI'
    cache_dir=check_dir(os.path.dirname(opts.get('experiment_path')),'caches')

    assert in_channels == 1, 'Currently only support single channel input'

    cropper = [RandCropByPosNegLabeld(keys=["image","label"], label_key='label', pos=1, neg=1, spatial_size=crop_size)] if is_avaible_size(crop_size) else []
    #cropper = [RandSpatialCropD(keys=['image','label'], roi_size=crop_size, random_size=False)] if is_avaible_size(crop_size) else []
    if phase == 'train':
        additional_transforms = [
            RandFlipD(keys=["image","label"], prob=augment_ratio, spatial_axis=[2]),
            RandRotateD(keys=["image","label"], range_x=math.pi/40, range_y=math.pi/40, range_z=math.pi/40, prob=augment_ratio, padding_mode='border'),
            Rand3DElasticD(keys=["image","label"], prob=augment_ratio, sigma_range=(5,10),
                           magnitude_range=(50,150), mode=["bilinear","nearest"], padding_mode='zeros')
        ]
    elif phase == 'valid':
        additional_transforms = []
    elif phase == 'test':
        cropper = None
        additional_transforms = []
    elif phase == 'test_wo_label':
        return UnsupervisedSegmentationDataset3D(
                    files_list,
                    orienter=Orientationd(keys='image', axcodes=orientation),
                    spacer=SpacingD(keys="image", pixdim=spacing),
                    resizer=None,
                    rescaler=ScaleIntensityRangePercentilesD(keys='image',lower=0.1, upper=99.9, b_min=0, b_max=1, clip=True),
                    cropper=None,
                    additional_transforms=[],
                    preload=0,
                    cache_dir=cache_dir,
                ).get_dataset()
    else:
        raise ValueError

    dataset = SupervisedSegmentationDataset3D(
        files_list,
        orienter=Orientationd(keys=['image','label'], axcodes=orientation),
        spacer=SpacingD(keys=["image","label"], pixdim=spacing),
        resizer=None,
        rescaler=ScaleIntensityRangePercentilesD(keys='image',lower=0.1, upper=99.9, b_min=0, b_max=1, clip=True),
        cropper=[CropForegroundD(keys=['image','label'],source_key='image',select_fn=lambda x: x > 0.45, margin=5),]+cropper,
        additional_transforms=additional_transforms,
        preload=preload,
        cache_dir=cache_dir,
    ).get_dataset()

    return dataset


def get_basis_rjh_tswi_dataset(files_list, phase, opts, spatial_size):
    spacing=opts.get('spacing', (0.66667, 0.66667, 1.34))
    in_channels=opts.get('input_nc', 3)
    preload=opts.get('preload', 0)
    augment_ratio=opts.get('augment_ratio', 0.4)
    orientation=opts.get('orientation', 'RAI')
    cache_dir=check_dir(os.path.dirname(opts.get('experiment_path')), 'caches')
    image_keys = opts.get('image_keys', ['image'])

    cropper = [
            SeparateCropSTSdataD(
                keys=image_keys+["mask"],
                mask_key="mask",
                label_key='label',
                crop_size=(32,32,16),
                # margin_size=(3, 3, 2),
                labels=[2, 1],
                flip_label=2,
                flip_axis=1,
            ),
            ExtractSTSlicesD(
                keys=image_keys+['mask'],
                mask_key='mask'
            ),
            SqueezeDimD(
                keys=image_keys+['mask'],
                dim=0
            ),
            AsChannelFirstD(
                keys=image_keys+['mask']
            )
        ]
    if spatial_size != (32, 32):
        cropper += [
            ResizeD(
                keys=image_keys+['mask'],
                spatial_size=spatial_size,
                mode=['bilinear', 'nearest']
            )
        ]

    if phase == 'train':
        additional_transforms = [
            RandAffineD(
                keys=image_keys+['mask'],
                spatial_size=spatial_size,
                prob=augment_ratio,
                rotate_range=[math.pi/30, math.pi/30],
                shear_range=[0.1, 0.1],
                translate_range=[1, 1],
                scale_range=[0.1, 0.1],
                mode=["bilinear", "nearest"],
                padding_mode=['reflection', 'zeros'],
                as_tensor_output=False
            )
        ]
    elif phase in ['valid', 'test']:
        additional_transforms = []
    elif phase == 'test_wo_label':
        raise NotImplementedError


    dataset = BasicClassificationDataset(
        files_list,
        loader=LoadNiftid(keys=image_keys+["mask"], dtype=np.float32),
        channeler=AddChannelD(keys=image_keys+['mask']),
        orienter=None, #Orientationd(keys=['image','mask'], axcodes=orientation),
        spacer=SpacingD(
            keys=image_keys+["mask"],
            pixdim=spacing,
            mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]
        ),
        rescaler=None,
        resizer=None,
        cropper=cropper,
        caster=CastToTyped(keys=image_keys, dtype=np.float32),
        to_tensor=ToTensord(keys=image_keys+["mask"]),
        is_supervised=True,
        dataset_type=CacheDataset,
        dataset_kwargs={'cache_rate':preload},
        additional_transforms=additional_transforms,
    ).get_dataset()

    return dataset


@CLASSIFICATION_DATASETS.register('2D', 'rjh_tswi_oneside',
    "/homes/clwang/Data/RJH/STS_tSWI/datalist_wi_mask@1229_2131-train.json")
def get_oneside_dataset(files_list, phase, opts):
    return get_basis_rjh_tswi_dataset(files_list, phase, opts, (32,32))


@CLASSIFICATION_DATASETS.register('2D', 'rjh_tswi_oneside_larger',
    "/homes/clwang/Data/RJH/STS_tSWI/datalist_wi_mask@1229_2131-train.json")
def get_oneside_dataset(files_list, phase, opts):
    return get_basis_rjh_tswi_dataset(files_list, phase, opts, (64,64))


@CLASSIFICATION_DATASETS.register('2D', 'rjh_tswi_multimodal',
    "/homes/clwang/Data/RJH/STS_tSWI/datalist_wi_mask@1229_2131-train.json")
def get_oneside_dataset(files_list, phase, opts):
    opts['image_keys'] = ['image1', 'image2']
    return get_basis_rjh_tswi_dataset(files_list, phase, opts, (32,32))

