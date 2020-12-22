import os
import numpy as np
from utils_cw import Print, check_dir

from monai_ex.data import CacheDataset, PersistentDataset
from monai_ex.transforms import *

from medlp.data_io import CLASSIFICATION_DATASETS, SEGMENTATION_DATASETS

@SEGMENTATION_DATASETS.register('3D','kits','/MRIData/kits19/data/train_data_list.json')
def get_kits_dataset(files_list, phase, opts):
    spacing=opts.get('spacing', [])
    winlevel=opts.get('winlevel', [-80,304])
    in_channels=opts.get('input_nc', 1)
    image_size=opts.get('image_size', None)
    crop_size=opts.get('crop_size', None)
    preload=opts.get('preload', 1.0)
    augment_ratio=opts.get('augment_ratio', 0.4)
    orientation=opts.get('orientation', 'LPI')
    cache_dir=check_dir(os.path.dirname(opts.get('experiment_path')),'caches')
    
    data_reader = LoadNiftid(keys=["image","label"], dtype=np.float32)

    if spacing:
        spacer = Spacingd(keys=["image","label"], pixdim=spacing)
    else:
        spacer = Lambdad(keys=["image","label"], func=lambda x : x)

    if crop_size is None or np.any(np.less_equal(crop_size,0)):
        Print('No cropping!', color='g')
        cropper = Lambdad(keys=["image","label"], func=lambda x : x)
    else:
        assert len(crop_size) == 3, f'Dim of crop_size must equal to 3, but got {crop_size}'
        cropper = RandCropByPosNegLabeld(keys=["image","label"], label_key='label', spatial_size=crop_size)

    normalizer = ScaleIntensityRanged(keys="image", a_min=winlevel[0], a_max=winlevel[1], b_min=0, b_max=1, clip=True)

    if phase == 'train':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=['image','label'], axcodes=orientation),
            normalizer,
            spacer,
            cropper,
            # Rand3DElasticd(keys=["image","label"], prob=augment_ratio, sigma_range=(5,10), 
            #                magnitude_range=(50,150), mode=["bilinear","nearest"], padding_mode='zeros'),
            RandAdjustContrastd(keys="image", prob=augment_ratio, gamma=(0.8,1.2)),
            RandGaussianNoised(keys="image", prob=augment_ratio, std=0.02),
            RandRotated(keys=["image","label"], range_x=5, range_y=5, range_z=5, prob=augment_ratio, padding_mode='zeros'),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=0),
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    elif phase == 'valid':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=['image','label'], axcodes=orientation),
            normalizer,
            spacer,
            cropper,
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    elif phase == 'test':
        pass
    
    if preload == 1.0:
        dataset_ = PersistentDataset(files_list, transform=transforms, cache_dir=cache_dir)
    else:
        dataset_ = CacheDataset(files_list, transform=transforms, cache_rate=preload)
    return dataset_
       
