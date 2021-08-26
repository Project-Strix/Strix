from typing import Dict, List, Hashable, Mapping, Optional, Sequence, Union

import torch
import numpy as np
from scipy import ndimage as ndi
from skimage import exposure

from monai_ex.config import KeysCollection
from monai_ex.utils import ensure_tuple, ensure_list, ensure_tuple_rep
from monai_ex.transforms import (
    generate_spatial_bounding_box,
    generate_pos_neg_label_crop_centers,
    map_binary_to_indices,
    Transform,
    MapTransform,
    Randomizable,
    SpatialCrop,
)

from medlp.models.rcnn.structures.bounding_box import BoxList
from utils_cw import remove_outlier

class CoordToBoxList(Transform):
    """
    Converts the input data to a BoxList without applying any other transformations.
    """

    def __call__(self, 
                 coord: Union[np.ndarray, list, tuple],
                 label: int, 
                 shape: Union[np.ndarray, list, tuple],
                 box_radius: Union[np.ndarray, list, tuple]) -> BoxList:
        
        boxes = [[coord[0]-box_radius[0], coord[1]-box_radius[1], 
                  coord[0]+box_radius[0], coord[1]+box_radius[1]]]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)
        # and labels
        labels = torch.tensor([label])

        # create a BoxList from the boxes
        boxlist = BoxList(boxes, shape, mode="xyxy")
        # add the labels to the boxlist
        boxlist.add_field("labels", labels)

        return boxlist


class CoordToBoxListd(MapTransform):
    """
    Dictionary-based wrapper of transform CoordToBoxList.
    """

    def __init__(self, 
                 keys: KeysCollection,
                 box_radius:  Union[np.ndarray, list, tuple], 
                 image_shape: Union[np.ndarray, list, tuple],
                 label_key: Optional[KeysCollection] = None,
                 
        ) -> None:
        """
        Args:
            keys: keys of the corresponding items to be transformed.
                See also: :py:class:`monai.transforms.compose.MapTransform`
        """
        super().__init__(keys)
        self.radius = box_radius
        self.image_shape = image_shape
        self.label_key = label_key
        self.converter = CoordToBoxList()

    def __call__(self, data: Mapping[Hashable, Union[np.ndarray, torch.Tensor]]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)
        label_data = 1 if self.label_key is None else d[self.label_key]
        
        for key in self.keys:
            d[key] = self.converter(d[key], label=label_data, shape=self.image_shape, box_radius=self.radius)
        return d


class DataLabelling(Transform):
    def __init__(self) -> None:
        """
        Args:
            to_onehot: whether convert labelling data to onehot format.

        """
        #self.to_onehot = to_onehot
    
    def __call__(self, img: np.ndarray) -> np.ndarray:
        input_ndim = img.squeeze().ndim # spatial ndim
        if input_ndim == 2:
            structure = ndi.generate_binary_structure(2, 1)
        elif input_ndim == 3:
            structure = ndi.generate_binary_structure(3, 1)
        else:
            raise ValueError('Currently only support 2D&3D data')
        
        channel_dim = None
        if input_ndim != img.ndim:
            channel_dim = img.shape.index(1)
            img = img.squeeze()

        ccs, num_features = ndi.label(img, structure=structure)
        
        if channel_dim is not None:
            return np.expand_dims(ccs, axis=channel_dim)

        return ccs

class DataLabellingD(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
    ) -> None:
        super().__init__(keys)
        self.converter = DataLabelling()

    def __call__(self, img: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(img)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key])
        return d


class Clahe(Transform):
    def __init__(self, kernel_size=None, clip_limit=0.01, nbins=256) -> None:
        self.kernel_size=kernel_size 
        self.clip_limit=clip_limit
        self.nbins=nbins

    def __call__(self, img: np.ndarray) -> np.ndarray:
        input_ndim = img.squeeze().ndim # spatial ndim
        assert input_ndim in [2, 3], 'Currently only support 2D&3D data'

        channel_dim = None
        if input_ndim != img.ndim:
            channel_dim = img.shape.index(1)
            img = img.squeeze()
        
        filter_img = exposure.equalize_adapthist(img, kernel_size=self.kernel_size, clip_limit=self.clip_limit, nbins=self.nbins)

        if channel_dim is not None:
            return np.expand_dims(filter_img, axis=channel_dim)
        else:
            return filter_img
        
class ClaheD(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        kernel_size=None, 
        clip_limit=0.01, 
        nbins=256
    ) -> None:
        super().__init__(keys)
        self.converter = Clahe()
        self.kernel_size = kernel_size
        self.clip_limit = clip_limit
        self.nbins = nbins

    def __call__(self, img: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(img)
        for idx, key in enumerate(self.keys):
            d[key] = self.converter(d[key])
        return d


class LabelToSeparateMask(Transform):
    """
    Different from monai's LabelToMask which transfrom select_labels to one non-zero mask.
    LabelToSeparateMask transform each label to target labels.
    """
    def __init__(  # pytype: disable=annotation-type-mismatch
        self,
        select_labels: Union[Sequence[int], int]
    ) -> None:  # pytype: disable=annotation-type-mismatch
        self.select_labels = ensure_tuple(select_labels)


    def __call__(
        self, img: np.ndarray, merge_channels: bool = False
    ) -> np.ndarray:
        raise NotImplementedError


class RandLabelToMask(Randomizable, Transform):
    """
    Convert labels to mask for other tasks. A typical usage is to convert segmentation labels
    to mask data to pre-process images and then feed the images into classification network.
    It can support single channel labels or One-Hot labels with specified `select_labels`.
    For example, users can select `label value = [2, 3]` to construct mask data, or select the
    second and the third channels of labels to construct mask data.
    The output mask data can be a multiple channels binary data or a single channel binary
    data that merges all the channels.

    Args:
        select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
            is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
            `select_labels` is the expected channel indices.
        merge_channels: whether to use `np.any()` to merge the result on channel dim. if yes,
            will return a single channel mask with binary data.

    """

    def __init__(  # pytype: disable=annotation-type-mismatch
        self,
        select_labels: Union[Sequence[int], int],
        merge_channels: bool = False,
    ) -> None:  # pytype: disable=annotation-type-mismatch
        self.select_labels = ensure_tuple(select_labels)
        self.merge_channels = merge_channels

    def randomize(self):
        self.select_label = self.R.choice(self.select_labels, 1)[0]

    def __call__(
        self, img: np.ndarray, select_label: Optional[Union[Sequence[int], int]] = None, merge_channels: bool = False
    ) -> np.ndarray:
        """
        Args:
            select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
                is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
                `select_labels` is the expected channel indices.
            merge_channels: whether to use `np.any()` to merge the result on channel dim. if yes,
                will return a single channel mask with binary data.
        """
        if select_label is None:         
            self.randomize()
        else:
            self.select_label = select_label

        if img.shape[0] > 1:
            data = img[[self.select_label]]
        else:
            data = np.where(np.in1d(img, self.select_label), True, False).reshape(img.shape)

        return np.any(data, axis=0, keepdims=True) if (merge_channels or self.merge_channels) else data

class RandLabelToMaskD(Randomizable, MapTransform):
    """
    Dictionary-based wrapper of :py:class:`RandLabelToMask`.

    Args:
        keys: keys of the corresponding items to be transformed.
            See also: :py:class:`monai.transforms.compose.MapTransform`
        select_labels: labels to generate mask from. for 1 channel label, the `select_labels`
            is the expected label values, like: [1, 2, 3]. for One-Hot format label, the
            `select_labels` is the expected channel indices.
        merge_channels: whether to use `np.any()` to merge the result on channel dim.
            if yes, will return a single channel mask with binary data.

    """

    def __init__(  # pytype: disable=annotation-type-mismatch
        self,
        keys: KeysCollection,
        select_labels: Union[Sequence[int], int],
        merge_channels: bool = False,
        cls_label_key: Optional[KeysCollection] = None,
        select_msk_label: Optional[int] = None, #! for tmp debug
    ) -> None:
        super().__init__(keys)
        self.select_labels = select_labels
        self.cls_label_key = cls_label_key
        self.select_label = select_msk_label
        self.converter = RandLabelToMask(select_labels=select_labels, merge_channels=merge_channels)

    def randomize(self):
        self.select_label = self.R.choice(self.select_labels, 1)[0]

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        if self.select_label is None:
            self.randomize()

        if self.cls_label_key is not None:
            label = d[self.cls_label_key]
            assert len(label) == len(self.select_labels), 'length of cls_label_key must equal to length of mask select_labels'

            if isinstance(label, (list, tuple)):
                label = { i:L for i, L in enumerate(label, 1)}
            elif isinstance(label, (int, float)):
                label = {1:label}
            assert isinstance(label, dict), 'Only support dict type label'
            
            d[self.cls_label_key] = label[self.select_label]

        for key in self.keys:
            d[key] = self.converter(d[key], select_label=self.select_label)

        return d


class RandCropSliceD(Randomizable, MapTransform):
    def __init__(
        self,
        keys,
        mask_key,
        mode,
        pos: float = 1.0,
        neg: float = 1.0,
        spatial_size=None,
        num_samples=1,
        axis=0
    ):
        super().__init__(keys)
        if pos < 0 or neg < 0:
            raise ValueError(f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        if mode not in ['single', 'cross', 'parallel']:
            raise ValueError("Cropping mode must be one of 'single, cross, parallel'")

        self.mask_key = mask_key
        self.mode = mode
        self.pos_ratio = pos / (pos + neg)
        self.spatial_size = spatial_size
        self.num_samples = num_samples
        self.axis = axis

    def randomize(
        self,
        mask: np.ndarray,
        fg_indices: Optional[np.ndarray] = None,
        bg_indices: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(self.spatial_size, default=mask.shape[1:])
        if fg_indices is None or bg_indices is None:
            fg_indices_, bg_indices_ = map_binary_to_indices(mask, image, self.image_threshold)
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, mask.shape[1:], fg_indices_, bg_indices_, self.R
        )

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)

        fg_indices, bg_indices = map_binary_to_indices(d[self.mask_key], None, 0)
        self.randomize(d[self.mask_key], fg_indices, bg_indices)

        results: List[np.ndarray] = list()
        if self.centers is not None:
            for center in self.centers:
                cropper = SpatialCrop(roi_center=tuple(center), roi_size=self.spatial_size)
                results.append(cropper(img))

        return results


class SeparateCropSTSdataD(MapTransform):
    def __init__(
        self,
        keys,
        mask_key,
        label_key=None,
        crop_size=None,
        margin_size=None,
        labels=[1, 2],
        flip_label=2,
        flip_axis=1,
        outlier_size=20,
    ):
        super(SeparateCropSTSdataD, self).__init__(keys)
        assert len(labels) == 2, 'Only separate two labels'
        self.mask_key = mask_key
        self.label_key = label_key
        self.crop_size = crop_size
        self.margin_size = margin_size
        self.labels = labels
        self.flip_label = flip_label
        self.flip_axis = flip_axis
        self.outlier_size = outlier_size

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        d = dict(data)
        mask_data = d[self.mask_key]
        bboxes = []
        labels = []
        for label in self.labels:
            if np.count_nonzero(mask_data == label) == 0:
                continue
            mask_data_ = remove_outlier(mask_data == label, outlier_size=self.outlier_size)
            bboxes.append(generate_spatial_bounding_box(mask_data_))
            labels.append(label)

        new_bboxes = []
        for bbox in bboxes:
            margin = [0, ] * len(bboxes[0][0])
            if self.crop_size:
                margin = np.add(
                    margin,
                    [(self.crop_size[i]-(end-start))/2 for i, (start, end) in enumerate(zip(bbox[0], bbox[1]))]
                )
            if self.margin_size:
                margin = np.add(
                    margin,
                    self.margin_size
                )
            bbox = [
                np.subtract(bbox[0], margin-0.1).round().astype(int),
                np.add(bbox[1], margin+0.1).round().astype(int)
            ]
            new_bboxes.append(bbox)

        results: List[Dict[Hashable, np.ndarray]] = [dict() for _ in new_bboxes]
        for key in data.keys():
            if key in self.keys:
                for i, (bbox, label) in enumerate(zip(new_bboxes, labels)):
                    roi = tuple([..., ]+[slice(s, e) for s, e in zip(bbox[0], bbox[1])])
                    if label == self.flip_label:
                        results[i][key] = np.flip(d[key][roi], axis=self.flip_axis)
                    else:
                        results[i][key] = d[key][roi]
            elif self.label_key is not None and key == self.label_key:
                # separate labels
                for i in range(len(results)):
                    results[i][key] = data[self.label_key][i]
            else:
                for i in range(len(results)):
                    results[i][key] = data[key]

        return results


class RandSelectSTSdataD(SeparateCropSTSdataD, Randomizable):
    """Random select one side SN region for training.
       If both side is available (i.e. label in [0,1]), random select.
       If only one side is available, select the available one.
       Design for specific dataset. DONOT USE!
    """
    def __init__(
        self,
        keys,
        mask_key,
        label_key=None,
        crop_size=None,
        margin_size=None,
        labels=[1, 2],
        flip_label=2,
        flip_axis=1,
        outlier_size=20,
    ):
        super().__init__(
            keys,
            mask_key,
            label_key,
            crop_size,
            margin_size,
            labels,
            flip_label,
            flip_axis,
            outlier_size
        )

    def randomize(self, results):
        labels = [result[self.label_key] for result in results]
        available_labels = list(filter(lambda x: x[1] in [0, 1], enumerate(labels)))
        if len(available_labels) == 2:
            return self.R.choice(results)
        elif len(available_labels) == 1:
            return results[available_labels[0]]
        else:
            raise NotImplementedError(f"Got unexpected labels: {labels}")

    def __call__(self, data: Mapping[Hashable, np.ndarray]) -> Dict[Hashable, np.ndarray]:
        results = super().__call__(data)
        return self.randomize(results)


class ExtractSTSlicesD(MapTransform):
    """Extract the slices between SN and RN.
    Design for specific dataset. DONOT USE!
    """
    def __init__(self, keys, mask_key, n_slices=3, rn_label=3, axial=(0, 1), outlier_size=20):
        """
        Args:
            keys ([type]): Keys to pick data for transformation.
            mask_key ([type]): Key to pick mask data.
            n_slices (int, optional): Extract ``n_slices`` slices. Defaults to 3.
            rn_label (int, optional): Label of RN. Defaults to 3.
            axial (tuple, optional): Axial dims of image. Defaults to (0, 1).
        """
        super(ExtractSTSlicesD, self).__init__(keys)
        self.mask_key = mask_key
        self.rn_label = rn_label
        self.axial = axial
        self.n_slices = n_slices
        self.outlier_size = outlier_size

    def __call__(self, data):
        d = dict(data)

        mask_ = remove_outlier(
            d[self.mask_key].squeeze() == self.rn_label,
            outlier_size=self.outlier_size
        )
        rn_z = np.any(mask_, axis=self.axial)

        mask_ = remove_outlier(
            np.logical_and(
                d[self.mask_key].squeeze() > 0,
                d[self.mask_key].squeeze() != self.rn_label
            ),
            outlier_size=self.outlier_size
        )
        sn_z = np.any(mask_, axis=self.axial)

        try:
            rn_zmin, rn_zmax = np.where(rn_z)[0][[0, -1]]
            sn_zmin, sn_zmax = np.where(sn_z)[0][[0, -1]]
        except:
            print('mask shape:', d[self.mask_key].squeeze().shape, 'unique:', np.unique(d[self.mask_key]))
            raise ValueError(
                "No nonzero mask is found!\n"
                f"Image path: {d['image_meta_dict']['filename_or_obj']}")

        for key in self.keys:
            if sn_zmin < rn_zmin or (sn_zmax+sn_zmin) < (rn_zmax+rn_zmin):
                # selected_slices = slice(rn_zmin-self.n_slices-1, rn_zmin-1)
                selected_slices = slice(rn_zmin-self.n_slices, rn_zmin)
            else:
                print(d['image_meta_dict']['filename_or_obj'])
                raise NotImplementedError(f'rn z range: {rn_zmin} {rn_zmax}, sn z range: {sn_zmin} {sn_zmax}')

            d[key] = d[key][..., selected_slices].copy()
            #print('Slice shape:', d[key].shape, )
            if 0 in list(d[key].shape):
                raise ValueError(d['image_meta_dict']['filename_or_obj'])

        return d


class ConcatModalityD(MapTransform):
    """Concat multi-modality data by given keys.
    """
    def __init__(self, keys, output_key, axis):
        super().__init__(keys)
        self.output_key = output_key
        self.axis = axis

    def __call__(self, data):
        d = dict(data)
        concat_data = np.concatenate([d[key] for key in self.keys], axis=self.axis)
        d[self.output_key] = concat_data

        return d