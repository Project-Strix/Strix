import os, sys, time, torch, random, tqdm
import numpy as np
#from torch.utils.data import Dataset
from utils_cw import Print, load_h5
import nibabel as nib

# dataio_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append( os.path.join(os.path.dirname(dataio_dir), 'utils') )
# sys.path.append( dataio_dir )
#from dataio import load_picc_data_once
from scipy.ndimage.morphology import binary_dilation
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import monai
from monai.config import IndexSelection, KeysCollection
from monai.data import CacheDataset, Dataset
from monai.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple, InterpolateMode
from monai.transforms.utils import generate_pos_neg_label_crop_centers
from monai.transforms import (
    LoadHdf5d,
    LoadNumpyd,
    LoadNiftid,
    AddChanneld,
    SqueezeDimd,
    RepeatChanneld,
    Compose,
    Zoomd,
    Spacingd,
    SpatialCrop,
    CenterSpatialCropd,
    RandCropByPosNegLabeld,
    RandSpatialCropd,
    RandRotated,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    Lambdad,
    ToTensord,
    MapTransform, 
    Randomizable,
    CastToTyped,
    ThresholdIntensityd,
    NormalizeIntensityd,
    RandGaussianNoised,
    DivisiblePadd,
)


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


class PICC_RandCropByPosNegLabeld(Randomizable, MapTransform):
    """
    Dictionary-based version :py:class:`monai.transforms.RandCropByPosNegLabel`.
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of dictionaries for all the cropped images.
    """
    def __init__( self,
            keys: KeysCollection,
            label_key: str,
            tip_key: str,
            spatial_size: Union[Sequence[int], int],
            tip: float = 0.3,
            pos: float = 0.5,
            neg: float = 0.5,
            num_samples: int = 1,
            return_label = True
        ) -> None:
        super().__init__(keys)
        self.label_key = label_key
        self.spatial_size = spatial_size
        if pos < 0 or neg < 0 or tip < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg} tip={tip}.")
        if pos + neg + tip == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0 and tip=0.")
        self.pos_ratio = pos / (pos + neg)
        self.tip_ratio = tip
        self.num_samples = num_samples
        self.image_key = None
        self.tip_key = tip_key
        self.image_threshold = 0.
        self.centers: Optional[List[List[np.ndarray]]] = None
        self.return_label = return_label
        self.crop_labels = []

    def correct_centers(self, center_coord, spatial_size, max_size):
        # Select subregion to assure valid roi
        valid_start = np.floor_divide(spatial_size, 2)
        valid_end = np.subtract(max_size + np.array(1), spatial_size / np.array(2)).astype(np.uint16)  # add 1 for random
        for i in range(len(valid_start)):  # need this because np.random.randint does not work with same start and end
            if valid_start[i] == valid_end[i]:
                valid_end[i] += 1
        
        for i, c in enumerate(center_coord):
            center_i = c
            if c < valid_start[i]:
                center_i = valid_start[i]
            if c >= valid_end[i]:
                center_i = valid_end[i] - 1
            center_coord[i] = center_i
        return center_coord

    def generate_pos_neg_label_crop_centers( self,
            label: np.ndarray,
            spatial_size,
            num_samples: int,
            pos_ratio: float,
            image: Optional[np.ndarray] = None,
            image_threshold: float = 0.0,
            rand_state: np.random.RandomState = np.random,
        ):
        max_size = label.shape[1:]
        spatial_size = fall_back_tuple(spatial_size, default=max_size)
        if not (np.subtract(max_size, spatial_size) >= 0).all():
            raise ValueError("proposed roi is larger than image itself.")

        centers = []
        # Prepare fg/bg indices
        if label.shape[0] > 1:
            label = label[1:]  # for One-Hot format data, remove the background channel
        label_flat = np.any(label, axis=0).ravel()  # in case label has multiple dimensions
        fg_indices = np.nonzero(label_flat)[0]
        if image is not None:
            img_flat = np.any(image > image_threshold, axis=0).ravel()
            bg_indices = np.nonzero(np.logical_and(img_flat, ~label_flat))[0]
        else:
            bg_indices = np.nonzero(~label_flat)[0]

        if not len(fg_indices) or not len(bg_indices):
            if not len(fg_indices) and not len(bg_indices):
                raise ValueError("no sampling location available.")
            pos_ratio = 0 if not len(fg_indices) else 1

        crop_labels = []
        for _ in range(num_samples):
            crop_label = "pos" if rand_state.rand() < pos_ratio else "neg"
            crop_labels.append(crop_label)
            indices_to_use = fg_indices if crop_label=="pos" else bg_indices
            random_int = rand_state.randint(len(indices_to_use))
            center = np.unravel_index(indices_to_use[random_int], label.shape)
            center = center[1:]
            # shift center to range of valid centers
            center_ori = [c for c in center]
            centers.append(self.correct_centers(center_ori, spatial_size, max_size))

        return centers, crop_labels

    def randomize( self, label: np.ndarray,
            image: np.ndarray = None, fg_coord: np.ndarray = None ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=label.shape[1:])
        if fg_coord is None:
            self.centers, crop_labels = self.generate_pos_neg_label_crop_centers(
                label, self.spatial_size, self.num_samples, self.pos_ratio, image, self.image_threshold, self.R
            )
            self.crop_labels = crop_labels
        else:
            self.centers = [ self.correct_centers(fg_coord, self.spatial_size, label.shape[1:])]*self.num_samples
            self.crop_labels = ["tip"]*self.num_samples

    def __call__(self, data):
        d = dict(data)
        self.crop_labels = []
        label = d[self.label_key]
        image = d[self.image_key] if self.image_key else None
        tip_coord = d.get(self.tip_key, None)

        tip_coord = tip_coord if self.tip_ratio > self.R.rand() and np.all(tip_coord>0) else None
        self.randomize(label, image, tip_coord)
        assert isinstance(self.spatial_size, tuple)
        assert self.centers is not None
        results = [dict() for _ in range(self.num_samples)]
        for key in data.keys():
            if key in self.keys:
                img = d[key]
                for i, center in enumerate(self.centers):
                    Print('Center:', center, 'ROI size:', self.spatial_size, color='r', verbose=False)
                    cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)
                    results[i][key] = cropper(img)
                    if self.return_label:
                        results[i]["crop_label"] = self.crop_labels[i]
            else:
                for i in range(self.num_samples):
                    results[i][key] = data[key]

        return results


class PICC_RandomCropDataset(Dataset):
    def __init__(self, data, labels, coords, phase, num_samples=1, in_channels=1, augment_ratio=0.3, 
                 crop_size=(64,64), downsample=1, random_type='balance', **kwargs):
        self.images = data
        self.labels = labels
        self.coords = coords
        self.phase = phase
        self.num_samples = num_samples
        self.crop_size = (crop_size,)*2 if isinstance(crop_size,int) else crop_size
        self.downsample = downsample
        self.augment_ratio = augment_ratio
        self.in_channels = in_channels
        self.classes = ['pos', 'neg', 'tip']
        
        assert random_type in ['gt', 'balance']
        self.random_type = random_type
        self.pos_bias = kwargs.get('pos_bias', 0)
        self.transpose = kwargs.get('transpose', None)
        self.dynamic_size = kwargs.get('dynamic_size', False)
        self.augment_num = kwargs.get('augment_num', 1)
        self.verbose = kwargs.get('verbose', False)

        if self.random_type == 'gt':
            self.tip_ratio = 0.5
            self.pos_ratio = 1.0
            self.neg_raito = 0.0
        elif self.random_type == 'balance':
            self.tip_ratio = 0.3
            self.pos_ratio = 0.5
            self.neg_raito = 0.5

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, roi, coord = self.images[index], self.labels[index], self.coords[index]

        Repeat_channel = RepeatChanneld(keys=["img","roi"], repeats=self.in_channels) if self.in_channels > 1  else \
                         Lambdad(keys=["img", "roi"], func=lambda x : x)

        transforms = Compose([
            AddChanneld(keys=["img", "roi"]),
            Repeat_channel,
            RandScaleIntensityd(keys="img",factors=(-0.01,0.01), prob=self.augment_ratio),
            PICC_RandCropByPosNegLabeld(
                keys=["img", "roi"], label_key="roi", tip_key="coord", 
                tip=self.tip_ratio, pos=self.pos_ratio, neg=self.neg_raito,
                spatial_size=self.crop_size, num_samples=self.num_samples
            ),
            RandRotated(keys=["img","roi"], range_x=10, range_y=10, prob=self.augment_ratio),
            RandFlipd(keys=["img","roi"], prob=self.augment_ratio, spatial_axis=[0,1]),
            RandRotate90d(keys=["img", "roi"], prob=self.augment_ratio, spatial_axes=[0,1]),
            ToTensord(keys=["img", "roi"])
        ])

        if isinstance(image, str):
            Print('Loading', os.path.basename(image), color='y', verbose=True)
            data, roi, coord = load_h5(image, keywords=['image', 'roi', 'coord'], transpose=self.transpose)
            image  = np.squeeze(data).astype(np.float32)
            roi  = np.squeeze(roi).astype(np.uint8)
        else:
            image = np.squeeze(image).astype(np.float32)
            roi = np.squeeze(roi).astype(np.uint8)
            coord = np.array(coord).astype(np.float32)

        if self.downsample > 1:
            image = image[::self.downsample,::self.downsample]
            roi = roi[::self.downsample,::self.downsample]
            coord = coord/self.downsample

        data = transforms({"img":image, "roi":roi, "coord":coord})
        
        return {"image":data[0]['img'], "label":self.classes.index(data[0]['crop_label'])}


def get_PICC_dataset(files_list, phase, spacing=[], in_channels=1, crop_size=(0,0),
                     preload=True, augment_ratio=0.4, downsample=1, verbose=False):

    cache_ratio = 1.0 if preload else 0.0
    all_data = all_roi = all_coord = files_list
    data_reader = LoadHdf5d(keys=["image","label","coord"], h5_keys=["image","roi","coord"], 
                            affine_keys=["affine","affine",None], dtype=[np.float32, np.int64, np.float32])
    input_data = all_data

    if spacing:
        spacer = Spacingd(keys=["image","label"], pixdim=spacing)
    else:
        spacer = Lambdad(keys=["image", "label"], func=lambda x : x)

    if in_channels > 1:
        repeater = RepeatChanneld(keys="image", repeats=in_channels)
    else:
        repeater = Lambdad(keys="image", func=lambda x : x)

    if crop_size is None or np.any(np.less_equal(crop_size,0)):
        Print('No cropping!', color='g')
        cropper = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        cropper = CenterSpatialCropd(keys=["image","label"], roi_size=crop_size, allow_pad=True)

    def debug(x):
        print("image type:", type(x), x.dtype, "image shape:", x.shape)
        return x
        nib.save(nib.Nifti1Image(x, np.eye(4)), f'./{str(time.time())}.nii.gz')
        return x
    debugger = Lambdad(keys=["image", "label"], func=debug)

    if phase == 'train':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            spacer,
            cropper,
            RandScaleIntensityd(keys="image",factors=(-0.01,0.01), prob=augment_ratio),
            RandRotated(keys=["image","label"], range_x=10, range_y=10, prob=augment_ratio),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[0]),
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            repeater,
            ToTensord(keys=["image", "label"]),
        ])
    elif phase == 'valid' or phase == 'test':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["image", "label"]),
            spacer,
            cropper,
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    dataset_ = CacheDataset(input_data, transform=transforms, cache_rate=cache_ratio)
    return dataset_


def get_RIB_dataset(files_list, phase, in_channels=1, preload=True, image_size=None, 
                    crop_size=None, augment_ratio=0.4, downsample=1, verbose=False):

    cache_ratio = 1.0 if preload else 0.0
    input_data = []
    for img, msk in files_list:
        input_data.append({"image":img, "label":msk})

    if image_size is None or np.any(np.less_equal(image_size,0)):
        Print('No image resize!', color='g')
        resizer = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        resizer = CenterSpatialCropd(keys=["image","label"], roi_size=image_size, allow_pad=True)

    if crop_size is None or np.any(np.less_equal(crop_size,0)):
        Print('No cropping!', color='g')
        cropper = Lambdad(keys=["image", "label"], func=lambda x : x)
    else:
        cropper = RandSpatialCropd(keys=["image", "label"], roi_size=crop_size, random_size=False)

    if phase == 'train':
        transforms = Compose([
            LoadNiftid(keys=["image","label"]),
            AddChanneld(keys=["image", "label"]),
            NormalizeIntensityd(keys="image"),
            ThresholdIntensityd(keys="label", threshold=1, above=False, cval=1),
            Zoomd(keys=["image", "label"], zoom=1/downsample, mode=[InterpolateMode.AREA,InterpolateMode.NEAREST], keep_size=False),
            RandScaleIntensityd(keys="image",factors=(-0.01,0.01), prob=augment_ratio),
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
            NormalizeIntensityd(keys=["image"]),
            ThresholdIntensityd(keys=["label"], threshold=1, above=False, cval=1),
            Zoomd(keys=["image", "label"], zoom=1/downsample, mode=[InterpolateMode.AREA,InterpolateMode.NEAREST], keep_size=False),
            resizer,
            cropper,
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    elif phase == 'test':
        transforms = Compose([
            LoadNiftid(keys=["image","label"]),
            AddChanneld(keys=["image", "label"]),
            NormalizeIntensityd(keys=["image"]),
            ThresholdIntensityd(keys=["label"], threshold=1, above=False, cval=1),
            Zoomd(keys=["image", "label"], zoom=1/downsample, mode=[InterpolateMode.AREA,InterpolateMode.NEAREST], keep_size=False),
            DivisiblePadd(keys=["image","label"], k=16),
            CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
            ToTensord(keys=["image", "label"])
        ])
    dataset_ = CacheDataset(input_data, transform=transforms, cache_rate=cache_ratio)
    return dataset_

