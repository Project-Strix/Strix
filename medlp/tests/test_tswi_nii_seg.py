import os
from medlp.data_io.rjh_dataset import get_rjh_tswi_seg_dataset
from utils_cw import get_items_from_file
import nibabel as nib
import numpy as np

files_list = "/homes/clwang/Data/RJH/RJ_data/preprocessed/labeled_data_list.json"

dataset_ = get_rjh_tswi_seg_dataset(get_items_from_file(files_list, format='json'),
                                    phase='train',
                                    crop_size=(0,0,0),
                                    preload=0,
                                    augment_ratio=1,
                                    )
print("Len:", len(dataset_))
for i, data in enumerate(dataset_):
    print(i)
    print(type(data['image']),  data['image'].shape, data['label'].shape)
    save_dir = '/homes/clwang/Data/RJH/RJ_data'
    nib.save( nib.Nifti1Image(data['image'].numpy().squeeze(), np.eye(4)), os.path.join(save_dir, f'{i}-img.nii.gz') )
    nib.save( nib.Nifti1Image(data['label'].numpy().squeeze().astype(np.uint8), np.eye(4)), os.path.join(save_dir, f'{i}-roi.nii.gz') )
    input('Continue?')