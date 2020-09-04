import os, sys, time, torch, random
import numpy as np
from torch.utils.data import Dataset
from utils_cw import Print, crop3D, load_h5
import nibabel as nib

# dataio_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append( os.path.join(os.path.dirname(dataio_dir), 'utils') )
# sys.path.append( dataio_dir )
#from dataio import load_picc_data_once
from scipy.ndimage.morphology import binary_dilation
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.config import IndexSelection, KeysCollection
import monai
from monai.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple
from monai.transforms.utils import generate_pos_neg_label_crop_centers
from monai.transforms import (
    AddChanneld,
    Compose,
    SpatialCrop, 
    RandCropByPosNegLabeld,
    RandRotated,
    RandFlipd,
    RandRotate90d,
    RandScaleIntensityd,
    ToTensord,
    MapTransform, 
    Randomizable
)

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


class RandomCropDataset(Dataset):
    def __init__(self, data, labels, coords, num_samples=1, n_classes=3, augment_ratio=0.3, 
                 crop_size=(64,64), downsample=1, random_type='balance', **kwargs):
        self.images = data
        self.labels = labels
        self.coords = coords
        self.num_samples = num_samples
        self.crop_size = (crop_size,)*2 if isinstance(crop_size,int) else crop_size
        self.downsample = downsample
        self.channels = 1
        self.augment_ratio = augment_ratio
        self.n_classes = n_classes
        self.classes = ['tip', 'pos', 'neg']
        
        assert random_type in ['gt', 'balance']
        self.random_type = random_type
        self.pos_bias = kwargs.get('pos_bias', 0)
        self.transpose = kwargs.get('transpose', None)
        self.dynamic_size = kwargs.get('dynamic_size', False)
        self.augment_num = kwargs.get('augment_num', 1)
        self.verbose = kwargs.get('verbose', False)

        if self.random_type == 'gt':
            tip_ratio = 0.5
            pos_ratio = 1.0
            neg_raito = 0.0
        elif self.random_type == 'balance':
            tip_ratio = 0.3
            pos_ratio = 0.5
            neg_raito = 0.5
        
        self.augmentations = Compose( [
            AddChanneld(keys=["img", "roi"]),
            RandScaleIntensityd(keys="img",factors=(-0.01,0.01), prob=augment_ratio),
            PICC_RandCropByPosNegLabeld(
                keys=["img", "roi"], label_key="roi", tip_key="coord", 
                tip=tip_ratio, pos=pos_ratio, neg=neg_raito,
                spatial_size=self.crop_size, num_samples=self.num_samples
            ),
            RandRotated(keys=["img","roi"], range_x=10, range_y=10, prob=augment_ratio),
            RandFlipd(keys=["img","roi"], prob=augment_ratio, spatial_axis=[0,1]),
            RandRotate90d(keys=["img", "roi"], prob=augment_ratio, spatial_axes=[0,1]),
            ToTensord(keys=["img", "roi"])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image, roi, coord = self.images[index], self.labels[index], self.coords[index]

        if isinstance(image, str):
            Print('Loading', os.path.basename(image), color='y', end='\n', verbose=self.verbose)
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

        data = self.augmentations({"img":image, "roi":roi, "coord":coord})
        
        return {"image":data[0]['img'], "label":self.classes.index(data[0]['crop_label'])}
 