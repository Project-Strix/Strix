import os
from medlp.data_io.picc_dataset import PICC_nii_seg_dataset
from utils_cw import get_items_from_file
import nibabel as nib
import numpy as np

files_list = "/homes/clwang/Data/picc/picc_seg_nii.json"

dataset_ = PICC_nii_seg_dataset(get_items_from_file(files_list, format='json'),
                                phase='train',
                                spacing=(0.3,0.3),
                                in_channels=1,
                                image_size=(1024,1024),
                                crop_size=None,
                                preload=0,
                                augment_ratio=1
                                )

for i, data in enumerate(dataset_):
    print(i)
    print(type(data['image']),  data['image'].shape)
    save_dir = '/homes/clwang/Data/picc'
    nib.save( nib.Nifti1Image(data['image'], np.eye(4)), os.path.join(save_dir, f'{i}.nii.gz') )