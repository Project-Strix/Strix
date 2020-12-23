import os, sys, time, torch, random, tqdm, math
import numpy as np
from utils_cw import Print, load_h5, check_dir
import nibabel as nib

from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from medlp.data_io import CLASSIFICATION_DATASETS, SEGMENTATION_DATASETS
from medlp.data_io.base_dataset.segmentation_dataset import SupervisedSegmentationDataset3D, UnsupervisedSegmentationDataset3D
from medlp.data_io.base_dataset.classification_dataset import SupervisedClassificationDataset3D
from medlp.utilities.utils import is_avaible_size
from medlp.utilities.transforms import DataLabellingD, RandLabelToMaskD

import monai
from monai.config import IndexSelection, KeysCollection
from monai.data import CacheDataset, Dataset, PersistentDataset
from monai.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple, InterpolateMode
from monai.transforms.utils import generate_pos_neg_label_crop_centers
from monai.transforms import *


@SEGMENTATION_DATASETS.register('3D','prostatex',
    "/homes/clwang/Data/RJH/RJ_data/preprocessed/labeled_data_list.json")
def get_prostatex_seg_dataset(files_list, phase, opts):
    #spacing=(0.5,0.5,3)
    in_channels=opts.get('input_nc', 1)
    crop_size=opts.get('crop_size', (256,256))
    preload=opts.get('preload', 0)
    augment_ratio=opts.get('augment_ratio',0.4)
    orientation='RPI'
    cache_dir=check_dir(opts.get('experiment_path'),'caches')

    assert in_channels == 1, 'Currently only support single channel input'

    cropper = CenterSpatialCropD(keys=["image","label"],roi_size=crop_size) if \
              is_avaible_size(crop_size) else None
    
    if phase == 'train':
        additional_transforms = [
            RandFlipD(keys=["image","label"], prob=augment_ratio, spatial_axis=[2]),
            RandRotateD(keys=["image","label"], range_x=math.pi/40, range_y=math.pi/40, prob=augment_ratio, padding_mode='zeros')
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
                    spacer=None,
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
        orienter=Orientationd(keys='image', axcodes=orientation),
        spacer=None,
        resizer=None,
        rescaler=NormalizeIntensityD(keys='image'),
        cropper=cropper,
        additional_transforms=additional_transforms,
        preload=preload,
        cache_dir=cache_dir,
    ).get_dataset()

    return dataset