from typing import Dict, List, Hashable, Mapping, Optional, Sequence, Union

import numpy as np

from monai_ex.config import KeysCollection
from monai_ex.utils import ensure_tuple
from monai_ex.transforms import (
    generate_pos_neg_label_crop_centers,
    Transform,
    MapTransform,
    Randomizable,
    SpatialCrop,
)


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
