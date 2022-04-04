from typing import Dict, List, Hashable, Mapping, Optional, Sequence, Union

import torch
import numpy as np
from scipy import ndimage as ndi

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
