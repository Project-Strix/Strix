import os, math
import numpy as np
from utils_cw import check_dir

from medlp.data_io import CLASSIFICATION_DATASETS, SEGMENTATION_DATASETS
from medlp.data_io.base_dataset.segmentation_dataset import SupervisedSegmentationDataset3D, UnsupervisedSegmentationDataset3D
from medlp.data_io.base_dataset.classification_dataset import BasicClassificationDataset
from medlp.utilities.utils import is_avaible_size
from medlp.utilities.transforms import (
    DataLabellingD,
    RandLabelToMaskD,
    SeparateCropSTSdataD,
    ExtractSTSlicesD
)

from monai_ex.data import CacheDataset, PersistentDataset
from monai_ex.transforms import *


@CLASSIFICATION_DATASETS.register('2D', 'jsph_nodule',
    "/homes/clwang/Data/jsph_lung/YHBLXA_YXJB/datalist.json")
def get_25d_dataset(files_list, phase, opts):
    # median reso: 0.70703125 z_reso: 1.5
    spacing=opts.get('spacing', (0.7, 0.7, 1.5))
    in_channels=opts.get('input_nc', 3)
    preload=opts.get('preload', 0)
    augment_ratio=opts.get('augment_ratio', 0.4)
    orientation=opts.get('orientation', 'RAI')
    cache_dir=check_dir(os.path.dirname(opts.get('experiment_path')), 'caches')
    image_keys = opts.get('image_keys', ['image'])
    mask_keys = opts.get('mask_keys', ['mask'])

    cropper = []

    dataset = BasicClassificationDataset(
        files_list,
        loader=LoadNiftiD(keys=image_keys+mask_keys, dtype=np.float32),
        channeler=AddChannelD(keys=image_keys+mask_keys),
        orienter=None, #Orientationd(keys=['image','mask'], axcodes=orientation),
        spacer=SpacingD(
            keys=image_keys+mask_keys,
            pixdim=spacing,
            mode=[GridSampleMode.BILINEAR, GridSampleMode.NEAREST]
        ),
        rescaler=None,
        resizer=None,
        cropper=cropper,
        caster=CastToTyped(keys=image_keys, dtype=np.float32),
        to_tensor=ToTensord(keys=image_keys+mask_keys),
        is_supervised=True,
        dataset_type=CacheDataset,
        dataset_kwargs={'cache_rate': preload},
        additional_transforms=additional_transforms,
    ).get_dataset()

    return dataset

