import os, math
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence
from utils_cw import Print

from monai.data import CacheDataset, PersistentDataset
from monai.transforms import *
from monai.transforms.compose import MapTransform
from monai.utils import ensure_list

class BasicClassificationDataset(object):
    def __init__(
        self, 
        files_list,
        loader,
        channeler,
        orienter,
        spacer,
        rescaler,
        resizer,
        cropper,
        caster,
        to_tensor,
        additional_transforms,
        verbose=False
    ):
        self.files_list = files_list
        self.verbose=verbose
        self.input_data = self.get_input_data()
        self.dataset = None

        self.transforms = ensure_list(loader)
        if channeler is not None:
            self.transforms += ensure_list(channeler)
        if orienter is not None:
            self.transforms += ensure_list(orienter)
        if spacer is not None:
            self.transforms += ensure_list(spacer)
        if rescaler is not None:
            self.transforms += ensure_list(rescaler)
        if resizer is not None:
            self.transforms += ensure_list(resizer)
        if cropper is not None:
            self.transforms += ensure_list(cropper)

        self.transforms += ensure_list(additional_transforms)

        if caster is not None:
            self.transforms += ensure_list(caster)
        if to_tensor is not None:
            self.transforms += ensure_list(to_tensor)

        self.transforms = Compose(self.transforms)
    
    def get_input_data(self):
        '''
        check input file_list format and existence.
        '''
        input_data = []
        for f in self.files_list:
            if isinstance(f, (list, tuple)): # Recognize f as ['image','label']
                assert os.path.exists(f[0]), f'Image file not exists: {f[0]}'
                assert os.path.exists(f[1]), f'Image file not exists: {f[1]}'
                input_data.append({"image":f[0], "label":f[1]})
            elif isinstance(f, dict):
                assert 'image' in f, f"File {f} doesn't contain keyword 'image'"
                assert os.path.exists(f['image']), f"File not exists: {f['image']}"
                input_data.append({"image":f['image'], "label":f['label']})
            else:
                raise ValueError(f'Not supported file_list format, Got {type(f)}')
            
        return input_data

    def get_dataset(self):
        return self.dataset


class ClassificationDataset2D(BasicClassificationDataset):
    def __init__(
        self, 
        files_list,
        loader: MapTransform = LoadPNGD(keys="image", grayscale=True),
        channeler: Optional[MapTransform] = AddChannelD(keys="image"),
        orienter: Optional[MapTransform] = OrientationD(keys='image', axcodes='LPI'),
        spacer: Optional[MapTransform] = SpacingD(keys="image", pixdim=(0.1,0.1)),
        rescaler: Optional[MapTransform] =  ScaleIntensityRangeD(keys="image", a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
        resizer: Optional[MapTransform] = ResizeD(keys="image", spatial_size=(512,512)),
        cropper: Optional[MapTransform] = RandSpatialCropD(keys="image", roi_size=(256,256), random_size=False),
        additional_transforms: Optional[Sequence] = None,
        caster: Optional[MapTransform] = CastToTypeD(keys=["image","label"], dtype=[np.float32, np.int64]),
        to_tensor: Optional[MapTransform] = ToTensorD(keys=["image","label"]),
        augment_ratio: Optional[float] = 0.5,
        preload: float = 1.0, 
        verbose: bool = False
    ):
        assert loader is not None, 'Loader cannot be None'
        
        # Give default 2D segmentation trasformations
        if additional_transforms is None: 
            additional_transforms = [
                RandAdjustContrastd(keys="image", prob=augment_ratio, gamma=(0.7,1.4)),
                RandRotated(keys="image", range_x=math.pi/18, range_y=math.pi/18, prob=augment_ratio),
                RandFlipd(keys="image", prob=augment_ratio, spatial_axis=[1])
            ]
        
        super().__init__(files_list, 
                         loader,
                         channeler,
                         orienter,
                         spacer,
                         rescaler,
                         resizer,
                         cropper,
                         caster,
                         to_tensor,
                         additional_transforms,
                         verbose
                         )

        self.dataset = CacheDataset(self.input_data, transform=self.transforms, cache_rate=preload)


class ClassificationDataset3D(BasicClassificationDataset):
    def __init__(
        self, 
        files_list,
        loader: MapTransform = LoadNiftid(keys="image", dtype=np.float32),
        channeler: Optional[MapTransform] = AddChanneld(keys="image"),
        orienter: Optional[MapTransform] = Orientationd(keys='image', axcodes='LPI'),
        spacer: Optional[MapTransform] = Spacingd(keys="image", pixdim=(1,1,1)),
        rescaler: Optional[MapTransform] =  ScaleIntensityRanged(keys="image", a_min=0, a_max=65535, b_min=0, b_max=1, clip=True),
        resizer: Optional[MapTransform] = ResizeWithPadOrCropd(keys="image", spatial_size=(512,512,256)),
        cropper: Optional[MapTransform] = RandSpatialCropD(keys="image", roi_size=(96,96,96), random_size=False),
        additional_transforms: Optional[Sequence] = None,
        caster: Optional[MapTransform] = CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
        to_tensor: Optional[MapTransform] = ToTensord(keys=["image","label"]),
        augment_ratio: float = 0.5,
        preload: float = 1.0,
        cache_dir: str = './',
        verbose: bool = False
    ):
        assert loader is not None, 'Loader cannot be None'
        
        # Give default 2D segmentation trasformations
        if additional_transforms is None: 
            additional_transforms = [
                RandAdjustContrastd(keys="image", prob=augment_ratio, gamma=(0.7,1.4)),
                RandGaussianNoised(keys="image", prob=augment_ratio, std=0.03),
                RandRotated(keys="image", range_x=math.pi/18, range_y=math.pi/18, prob=augment_ratio),
                RandFlipd(keys="image", prob=augment_ratio, spatial_axis=[1])
            ]
        
        super().__init__(files_list, 
                         loader,
                         channeler,
                         orienter,
                         spacer,
                         rescaler,
                         resizer,
                         cropper,
                         caster,
                         to_tensor,
                         additional_transforms,
                         verbose
                         )

        if preload == 1.0:
            self.dataset = PersistentDataset(self.input_data, transform=self.transforms, cache_dir=cache_dir)
        else:
            self.dataset = CacheDataset(self.input_data, transform=self.transforms, cache_rate=preload)
        