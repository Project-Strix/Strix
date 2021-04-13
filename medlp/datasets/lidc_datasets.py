import os, math
import numpy as np
from numpy.lib.npyio import savez_compressed
from utils_cw import check_dir

from medlp.data_io.dataio import SEGMENTATION_DATASETS
from medlp.data_io.base_dataset.segmentation_dataset import BasicSegmentationDataset
from medlp.utilities.utils import is_avaible_size
from medlp.utilities.transforms import (
    DataLabellingD,
    RandLabelToMaskD,
    SeparateCropSTSdataD,
    ExtractSTSlicesD,
    RandCrop2dByPosNegLabelD,
)

from monai_ex.data import CacheDataset, PersistentDataset
from monai_ex.transforms import *
from monai_ex.engines.utils import CustomKeys


@SEGMENTATION_DATASETS.register('3D', 'lidc',
    "/homes/clwang/Data/LIDC-IDRI-Crops-Norm/train-yrq.json")
def get_3d_dataset(files_list, phase, opts):
    return get_lung_dataset(files_list, phase, opts, (64,64,16))


def get_lung_dataset(files_list, phase, opts, spatial_size):
    in_channels = opts.get('input_nc', 1)
    preload = opts.get('preload', 0)
    augment_ratio = opts.get('augment_ratio', 0.4)
    spacing = opts.get('spacing', (0.7, 0.7, 1.25))
    image_keys = opts.get('image_keys', ['image'])
    mask_keys = opts.get('mask_keys', ['label'])

    if phase == 'train':
        additional_transforms = [
            RandRotate90D(
                keys=image_keys+mask_keys,
                prob=augment_ratio,
            ),
            RandAdjustContrastD(
                keys=image_keys,
                prob=augment_ratio,
                gamma=(0.9, 1.1)
            ),
            # RandAffineD(
            #     keys=image_keys+mask_keys,
            #     prob=augment_ratio,
            #     rotate_range=[math.pi/30, math.pi/30],
            #     shear_range=[0.1, 0.1],
            #     translate_range=[1, 1],
            #     scale_range=[0.1, 0.1],
            #     mode=["bilinear", "nearest"],
            #     padding_mode=['reflection', 'zeros'],
            #     as_tensor_output=False
            # )
        ]
    elif phase in ['valid', 'test']:
        additional_transforms = []
    elif phase == 'test_wo_label':
        raise NotImplementedError

    # cache_dir = check_dir(os.path.dirname(opts.get('experiment_path')), 'cache_dir')

    dataset = BasicSegmentationDataset(
        files_list,
        loader=LoadNiftiD(keys=image_keys+mask_keys, dtype=np.float32),
        channeler=AddChannelD(keys=image_keys+mask_keys),
        orienter=None,  # Orientationd(keys=['image','mask'], axcodes=orientation),
        spacer=SpacingD(keys=image_keys+mask_keys, pixdim=spacing),
        rescaler=None,
        resizer=None,
        cropper=ResizeWithPadOrCropD(keys=image_keys+mask_keys, spatial_size=spatial_size),
        caster=CastToTyped(keys=image_keys+mask_keys, dtype=[np.float32, np.int64]),
        to_tensor=ToTensord(keys=image_keys+mask_keys),
        is_supervised=True,
        dataset_type=CacheDataset,
        dataset_kwargs={'cache_rate': preload},
        additional_transforms=additional_transforms,
    ).get_dataset()

    return dataset
