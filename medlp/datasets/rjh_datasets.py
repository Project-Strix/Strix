import os
import math
import numpy as np
from utils_cw import check_dir

from medlp.data_io import CLASSIFICATION_DATASETS
from medlp.data_io.base_dataset.classification_dataset import BasicClassificationDataset
from medlp.utilities.utils import is_avaible_size
from medlp.utilities.transforms import (
    DataLabellingD,
    SeparateCropSTSdataD,
    ExtractSTSlicesD
)

from monai_ex.data import CacheDataset
from monai_ex.transforms import *


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

