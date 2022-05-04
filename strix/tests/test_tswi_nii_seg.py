import os
from monai_ex.transforms import LabelMorphologyD
from strix.data_io import DATASET_MAPPING

from utils_cw import get_items_from_file
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

# dataset_ =  get_rjh_tswi_cls_dataset(
#     input_data,
#     phase='train',
#     spacing=(0.667,0.667,1.34),
#     winlevel=None,
#     in_channels=1,
#     crop_size=None,
#     preload=0,
#     augment_ratio=1,
#     orientation='RAI',
#     cache_dir='./',
#     verbose=False
# )

opts = {
    'spacing':(0.667,0.667,1.34),
    'in_channel':1,
    'crop_size':(96,96,64),
    'preload':0,
    'augment_ratio':1,
    'orientation': 'RAI',
    'cache_dir':'./',
    'experiment_path': './'
    }

arguments = {
    'files_list': get_items_from_file("/homes/clwang/Data/RJH/RJ_data/tSWI_preprocessed/train_datalist_sn.json", format='json'),
    'phase': 'train',
    'opts': opts
}

dataset_ = DATASET_MAPPING['segmentation']['3D']['rjh_tswi_v2']["FN"](**arguments)

print("Len:", len(dataset_))
for i, data in enumerate(dataset_):
    img = data[0]['image']
    msk = data[0]['label']
    print(i, img.shape, 'label:', msk.shape)
   
    save_dir = '/homes/clwang/Data/RJH'
    nib.save( nib.Nifti1Image(img.numpy().squeeze(), np.eye(4)), os.path.join(save_dir, f'{i}-img.nii.gz') )
    nib.save( nib.Nifti1Image(msk.numpy().squeeze().astype(np.uint8), np.eye(4)), os.path.join(save_dir, f'{i}-roi.nii.gz') )
    input('Continue?')