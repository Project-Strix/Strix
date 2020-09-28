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
from monai.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple
from monai.transforms.utils import generate_pos_neg_label_crop_centers
from monai.transforms import (
    LoadHdf5d,
    LoadNumpyd,
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

    if np.any(np.less_equal(crop_size,0)):
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
            SqueezeDimd(keys=["label"]), #! to fit the demand of CE
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
            SqueezeDimd(keys=["label"]),
            ToTensord(keys=["image", "label"])
        ])
    dataset_ = CacheDataset(input_data, transform=transforms, cache_rate=cache_ratio)
    return dataset_


def get_RIB_dataset(files_list, phase, in_channels=1, perload=True, crop_size=(0,0),
                    augment_ratio=0.4, downsample=1, verbose=False):
    raise NotImplementedError
    if perload:
        all_data, all_roi = load_picc_h5_data_once(files_list, h5_keys=['image', 'roi'], transpose=None)
        data_reader = LoadNumpyd(keys=["image","label"])
        input_data = {"image":all_data,"label":all_roi}
    else:
        all_data = all_roi = all_coord = files_list    
        input_data = all_data
        data_reader = LoadHdf5d(keys=["image","label"], h5_keys=["image","roi","coord","affine"])
        input_data = all_data

    if phase == 'train':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["img", "roi"]),
            Zoomd(keys=["img", "roi"], zoom=1/downsample, keep_size=False),
            RandScaleIntensityd(keys="img",factors=(-0.01,0.01), prob=augment_ratio),
            RandSpatialCropd(keys=["img", "roi"], roi_size=crop_size, random_size=False),
            RandRotated(keys=["img","roi"], range_x=10, range_y=10, prob=augment_ratio),
            #RandFlipd(keys=["img","roi"], prob=augment_ratio, spatial_axis=[1]),
            #RandRotate90d(keys=["img", "roi"], prob=augment_ratio, spatial_axes=[0,1]),
            ToTensord(keys=["img", "roi"])
        ])
    elif phase == 'valid' or phase == 'test':
        transforms = Compose([
            data_reader,
            AddChanneld(keys=["img", "roi"]),
            Zoomd(keys=["img", "roi"], zoom=1/downsample, keep_size=False),
            ToTensord(keys=["img", "roi"])
        ])
    dataset_ = CacheDataset(input_data, transform=transforms, cache_rate=0.5)
    return dataset_

from skimage import transform, util, exposure
class RIB_dataset(Dataset):
    def __init__(self, raw_list, mask_list, crop_size=(512,512), mode='train', augmentation_prob=0.6):

        """Initializes image paths and preprocessing module."""
        self.images = raw_list
        self.masks = mask_list
        self.crop_size = crop_size
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(
            self.mode, len(self.images)))


    def random_crop2D(self, img, mask, size, verbose=False):
        xrange = abs(img.shape[0] - size[0])
        yrange = abs(img.shape[1] - size[1])
        xstart = random.randint(0, xrange) if xrange > 0 else 0
        ystart = random.randint(0, yrange) if yrange > 0 else 0
        crop_start = [xstart, ystart]
        if verbose:
            print('random crop at {} with size {}'.format(crop_start, size))

        img_cropped = img[xstart:xstart + size[0], ystart:ystart + size[1]]
        mask_cropped = mask[xstart:xstart + size[0], ystart:ystart + size[1]]

        return img_cropped, mask_cropped

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image = self.images[index]
        mask = self.masks[index]
        if isinstance(image, str):
            image = nib.load(image).get_fdata()  # 图像90°向右翻转, dtype=float64
            mask = nib.load(mask).get_fdata() 
        else:
            image = np.squeeze(image).astype(np.float32)
            mask = np.squeeze(mask).astype(np.int64)
         
        data = self.transforms({"img":image, "roi":mask})
        
        return {"image":data['img'], "label":data['roi']}

        # image = transform.resize(image, (2048,2048))
        # mask = transform.resize(mask, (2048,2048)) 

        # image = F.adjust_contrast(image, 2)

        # data augmentation
        if (self.mode == 'train') and random.random() <= self.augmentation_prob:
            rotate_angle = random.randint(-10, 10)
            image = transform.rotate(image, rotate_angle)
            mask = transform.rotate(mask, rotate_angle)
        if (self.mode == 'train') and random.random() <= self.augmentation_prob: 
            # horizontal flip
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1) 

        # # 对比度调整
        # image = exposure.adjust_log(image, 1)

        if (self.mode == 'train') and random.random() <= self.augmentation_prob:
            image = util.random_noise(image,mode='gaussian') 

        # 对比度均衡
        # image = np.repeat(np.expand_dims(image,axis=0),3,axis=0)
        # image = torch.tensor(image)
        # image = np.array(F.adjust_contrast(image, 2))[0, ...]

        img_cropped, mask_cropped = self.random_crop2D(image, mask, (768, 768))
        img_cropped, mask_cropped = img_cropped[::2, ::2], mask_cropped[::2, ::2]  # downsample for tmp exp

        img_cropped = np.expand_dims(img_cropped, axis=0)
        mask_cropped = np.expand_dims(mask_cropped, axis=0)

        return {"image":img_cropped.astype(np.float32), "label":mask_cropped.astype(np.int8)}

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.images)