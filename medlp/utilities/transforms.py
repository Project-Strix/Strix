from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import numpy as np

from monai.config import IndexSelection, KeysCollection
from monai.transforms.compose import Transform, MapTransform

from medlp.models.rcnn.structures.bounding_box import BoxList

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