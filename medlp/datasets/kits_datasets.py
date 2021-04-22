import os
import math
import numpy as np
from utils_cw import Print, check_dir

from monai_ex.data import CacheDataset, PersistentDataset
from monai_ex.transforms import *

from medlp.data_io import SEGMENTATION_DATASETS
from medlp.data_io.base_dataset.classification_dataset import BasicClassificationDataset


def get_kits_dataset(files_list, phase, opts):
    spacing=opts.get('spacing', [1,1,1])
    winlevel=opts.get('winlevel', [-80,304])
    in_channels=opts.get('input_nc', 1)
    image_size=opts.get('image_size', None)
    crop_size=opts.get('crop_size', (32,64,64))
    preload=opts.get('preload', 1.0)
    augment_ratio=opts.get('augment_ratio', 0.4)
    orientation=opts.get('orientation', 'LPI')
    cache_dir=check_dir(os.path.dirname(opts.get('experiment_path')),'caches')

    data_reader = LoadImageD(keys=["image", "label"], dtype=np.float32)

    if spacing:
        spacer = Spacingd(keys=["image", "label"], pixdim=spacing)
    else:
        spacer = Lambdad(keys=["image", "label"], func=lambda x : x)

    if crop_size is None or np.any(np.less_equal(crop_size,0)):
        Print('No cropping!', color='g')
        cropper = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        assert len(crop_size) == 3, f'Dim of crop_size must equal to 3, but got {crop_size}'
        cropper = RandCropByPosNegLabeld(keys=["image", "label"], label_key='label', spatial_size=crop_size)

    normalizer = ScaleIntensityRanged(keys="image", a_min=winlevel[0], a_max=winlevel[1], b_min=0, b_max=1, clip=True)

    if phase == 'train':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=['image', 'label'], axcodes=orientation),
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
            Orientationd(keys=['image', 'label'], axcodes=orientation),
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

@SEGMENTATION_DATASETS.register('3D', 'kitsHR', "/homes/clwang/Data/kits19/train_data_list_HR.json")
@SEGMENTATION_DATASETS.register('3D', 'kits', "/homes/clwang/Data/kits19/train_data_list.json")
def kits_dataset(files_list, phase, opts):
    orientation = 'RPI'
    spacing = (1, 1, 1)
    winlevel = [-79, 304]
    mean_, std_ = 100.93, 76.9
    crop_size = (160, 160, 128)  # (128, 128, 64)
    augment_ratio = opts.get('augment_ratio', 0.4)
    cache_dir = check_dir(
        os.path.dirname(opts.get('experiment_path')), 'caches'
    )

    cropper = [
        RandCropByPosNegLabelD(
            keys=["image", "label"], label_key='label', pos=2.0, spatial_size=crop_size),
        ResizeWithPadOrCropD(
            keys=["image", "label"], spatial_size=crop_size, mode="constant")
    ]
    if phase == 'train':
        additional_transforms = [
            RandAdjustContrastd(
                keys="image", prob=augment_ratio, gamma=(0.8, 1.2)
            ),
            RandGaussianNoised(
                keys="image", prob=augment_ratio, std=0.02
            ),
            RandRotated(
                keys=["image", "label"],
                range_x=math.pi/30, range_y=math.pi/30, range_z=math.pi/30,
                prob=augment_ratio, padding_mode='zeros'
            ),
            RandFlipd(
                keys=["image", "label"],
                prob=augment_ratio, spatial_axis=0
            ),
        ]
    elif phase == 'valid':
        additional_transforms = []
    else:
        cropper = None
        additional_transforms = []

    dataset = BasicClassificationDataset(
        files_list,
        loader=LoadImageD(
            keys=["image", "label"], dtype=np.float32
        ),
        channeler=AddChannelD(keys=['image', 'label']),
        orienter=Orientationd(
            keys=['image', 'label'], axcodes=orientation
        ),
        spacer=SpacingD(
            keys=["image", "label"],
            pixdim=spacing,
            mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]
        ),
        rescaler=[
            ClipIntensityD(keys="image", cmin=winlevel[0], cmax=winlevel[1]),
            NormalizeIntensityD(keys="image", subtrahend=mean_, divisor=std_)
        ],
        resizer=None,
        cropper=cropper,
        caster=CastToTyped(
            keys=["image", "label"],
            dtype=[np.float32, np.int64]
        ),
        to_tensor=ToTensord(keys=["image", "label"]),
        is_supervised=True,
        dataset_type=PersistentDataset,
        dataset_kwargs={'cache_dir': cache_dir},
        additional_transforms=additional_transforms
    ).get_dataset()

    return dataset
