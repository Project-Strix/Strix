import os
from medlp.data_io.rjh_dataset import get_rjh_tswi_seg_dataset
from medlp.data_io.segmentation_dataset import SegmentationDataset3D
from medlp.utilities.transforms import LabelMorphologyd
from monai.transforms import *

from utils_cw import get_items_from_file
import nibabel as nib
import numpy as np


files_list = "/homes/clwang/Data/RJH/RJ_data/preprocessed/labeled_data_list.json"

additional_trans = [LabelMorphologyd(keys='label', mode='dilation', radius=1)]
                    #Rand3DElasticD(keys=["image","label"], prob=1, sigma_range=(5,10),magnitude_range=(50,150), mode=["bilinear","nearest"], padding_mode='zeros')]

dataset_ = SegmentationDataset3D(
    get_items_from_file(files_list, format='json'),
    orienter=Orientationd(keys=['image','label'], axcodes="RAI"),
    spacer=SpacingD(keys=["image","label"], pixdim=(0.667,0.667,1.34), mode=[GridSampleMode.BILINEAR,GridSampleMode.NEAREST]),
    resizer=None,
    rescaler=NormalizeIntensityD(keys='image'),
    cropper=RandCropByPosNegLabeld(keys=["image","label"], label_key='label', pos=0, neg=1, spatial_size=(96,96,64)),
    additional_transforms=additional_trans,
    preload=0,
    cache_dir='./',
).get_dataset()

print("Len:", len(dataset_))
for i, data in enumerate(dataset_):
    print(i)
    print(type(data[0]['image']),  data[0]['image'].shape, data[0]['label'].shape)
    save_dir = '/homes/clwang/Data/RJH'
    nib.save( nib.Nifti1Image(data[0]['image'].numpy().squeeze(), np.eye(4)), os.path.join(save_dir, f'{i}-img.nii.gz') )
    nib.save( nib.Nifti1Image(data[0]['label'].numpy().squeeze().astype(np.uint8), np.eye(4)), os.path.join(save_dir, f'{i}-roi.nii.gz') )
    input('Continue?')