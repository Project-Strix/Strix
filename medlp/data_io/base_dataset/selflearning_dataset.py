import os
from abc import ABC, abstractmethod
from typing import Callable, Optional, Sequence
from utils_cw import Print

from monai_ex.data import CacheDataset, PersistentDataset
from monai_ex.transforms import *
from monai_ex.utils import ensure_list

class BasicSelflearningDataset(object):
    def __init__(
        self, 
        files_list,
        loader,
        channeler,
        orienter,
        repeater,
        spacer,
        rescaler,
        resizer,
        cropper,
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
            self.transforms.append(orienter)
        if repeater is not None:
            self.transforms.append(repeater)
        if spacer is not None:
            self.transforms.append(spacer)
        if rescaler is not None:
            self.transforms.append(rescaler)
        if resizer is not None:
            self.transforms.append(resizer)
        if cropper is not None:
            self.transforms.append(cropper)

        self.transforms += additional_transforms
        self.transforms = Compose(self.transforms)
    
    def get_input_data(self):
        '''
        check input file_list format and existence.
        '''
        input_data = []
        for f in self.files_list:
            if isinstance(f, str): # Recognize f as a file
                assert os.path.exists(f), f'File not exists: {f}'
                input_data.append({"image":f, "label":f})
            elif isinstance(f, dict):
                assert 'image' in f, f"File {f} doesn't contain keyword 'image'"
                assert os.path.exists(f['image']), f"File not exists: {f['image']}"
                input_data.append({"image":f['image'], "label":f['image']})
        return input_data

    def get_dataset(self):
        return self.dataset


class SelflearningDataset2D(BasicSelflearningDataset):
    def __init__(
        self, 
        files_list,
        loader: MapTransform = LoadPNGExd(keys=["image","label"], grayscale=True),
        channeler: Optional[MapTransform] = AddChanneld(keys=["image", "label"]),
        orienter: Optional[MapTransform] = Orientationd(keys=['image','label'], axcodes='LPI'),
        repeater: Optional[MapTransform] = RepeatChanneld(keys="image", repeats=3),
        spacer: Optional[MapTransform] = Spacingd(keys=["image","label"], pixdim=(0.1,0.1)),
        rescaler: Optional[MapTransform] =  ScaleIntensityRanged(keys=["image","label"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
        resizer: Optional[MapTransform] = ResizeWithPadOrCropd(keys=["image","label"], spatial_size=(768,768)),
        cropper: Optional[MapTransform] = RandSpatialCropd(keys=["image", "label"], roi_size=(512,512), random_size=False), 
        additional_transforms: Optional[Sequence] = None,
        augment_ratio: float = 0.5,
        preload: float = 1.0, 
        verbose: bool = False
    ):
        assert loader is not None, 'Loader cannot be None'
        
        # Give default 2D self-learning trasformations
        if additional_transforms is None: 
            additional_transforms = [
                RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[1]),
                adaptor(RandNonlinear(prob=augment_ratio), "image"),
                adaptor(RandLocalPixelShuffle(prob=augment_ratio, num_block_range=(9000,10000)), "image"),
                RandomSelect([
                    adaptor(RandImageInpainting(prob=1, num_block_range=(3,5)), 'image'),
                    adaptor(RandImageOutpainting(prob=1, num_block_range=(6,9)), 'image'),
                ], prob=augment_ratio),
                CastToTyped(keys=["image","label"], dtype=[np.float32, np.float32]),
                ToTensord(keys=["image","label"])
            ]

        super().__init__(files_list, 
                         loader,
                         channeler,
                         orienter,
                         repeater,
                         spacer,
                         rescaler,
                         resizer,
                         cropper,
                         additional_transforms,
                         verbose
                         )
        
        self.dataset = CacheDataset(self.input_data, transform=self.transforms, cache_rate=preload)

       
class SelflearningDataset3D(BasicSelflearningDataset):
    def __init__(
        self, 
        files_list,
        loader: MapTransform = LoadNiftid(keys=["image","label"], dtype=np.float32),
        channeler: Optional[MapTransform] = AddChanneld(keys=["image", "label"]),
        orienter: Optional[MapTransform] = Orientationd(keys=['image','label'], axcodes='LPI'),
        repeater: Optional[MapTransform] = RepeatChanneld(keys="image", repeats=3),
        spacer: Optional[MapTransform] = Spacingd(keys=["image","label"], pixdim=(0.1,0.1,0.1)),
        rescaler: Optional[MapTransform] =  ScaleIntensityRanged(keys=["image","label"], a_min=0, a_max=255, b_min=0, b_max=1, clip=True),
        resizer: Optional[MapTransform] = ResizeWithPadOrCropd(keys=["image","label"], spatial_size=(256,256,256)),
        cropper: Optional[MapTransform] = RandSpatialCropd(keys=["image", "label"], roi_size=(96,96,96), random_size=False), 
        additional_transforms: Optional[Sequence] = None,
        augment_ratio: float = 0.5,
        preload: float = 1.0, 
        cache_dir: str = './',
        verbose: bool = False
    ):
        assert loader is not None, 'Loader cannot be None'

        # Give default 3D self-learning trasformations
        if additional_transforms is None: 
            additional_transforms = [
                RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[1]),
                adaptor(RandNonlinear(prob=augment_ratio), "image"),
                adaptor(RandLocalPixelShuffle(prob=augment_ratio, num_block_range=(9000,10000)), "image"),
                RandomSelect([
                    adaptor(RandImageInpainting(prob=1, num_block_range=(3,5)), 'image'),
                    adaptor(RandImageOutpainting(prob=1, num_block_range=(6,9)), 'image'),
                ], prob=augment_ratio),
                CastToTyped(keys=["image","label"], dtype=[np.float32, np.float32]),
                ToTensord(keys=["image","label"])
            ]
        
        super().__init__(files_list, 
                         loader,
                         channeler,
                         orienter,
                         repeater,
                         spacer,
                         rescaler,
                         resizer,
                         cropper,
                         additional_transforms,
                         verbose
                         )
        
        if preload == 1.0:
            self.dataset = PersistentDataset(files_list, transform=self.transforms, cache_dir=cache_dir)
        else:
            self.dataset = CacheDataset(files_list, transform=self.transforms, cache_rate=preload)


    
