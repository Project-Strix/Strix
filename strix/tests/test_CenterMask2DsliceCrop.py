import os
import numpy as np
from utils_cw.augmentations import Compose
from utils_cw import get_items_from_file
import nibabel as nib
from monai_ex.data import DataLoader, Dataset
from monai_ex.transforms import *


file = r"/homes/clwang/Data/jsph_lung/YHBLXA_YXJB/datalist-train.json"
data_list = get_items_from_file(file, format='json')

cropper = CenterMask2DSliceCropD(
    keys=['image', 'mask'],
    mask_key='mask',
    roi_size=32,
    crop_mode='cross',
    center_mode='maximum',
    z_axis=2,
)


for i, data in enumerate(data_list):
    image = nib.load(data['image']).get_fdata()[np.newaxis, ...]
    mask = nib.load(data['mask']).get_fdata()[np.newaxis, ...]

    input_data = {'image': image, 'mask': mask}
    vol = cropper(input_data)
    print('Crop shape:', vol['image'].shape)
    nib.save( nib.Nifti1Image(vol['image'], np.eye(4)), f'/homes/clwang/Data/{i}testcrop.nii.gz' )
    nib.save( nib.Nifti1Image(vol['mask'], np.eye(4)), f'/homes/clwang/Data/{i}testmask.nii.gz' )