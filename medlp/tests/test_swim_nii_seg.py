import os
from medlp.data_io.rjh_dataset import get_rjh_swim_seg_dataset
#from medlp.data_io.base_dataset.segmentation_dataset import SegmentationDataset3D
from medlp.utilities.transforms import LabelMorphologyD
from monai.transforms import *

from utils_cw import get_items_from_file
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


files_list = "/homes/clwang/Data/RJH/RJ_data/preprocessed/labeled_data_list.json"
files_list = "/homes/clwang/Data/RJH/STS_tSWI/datalist_wi_mask@1124_1034.json"
files_list = "/homes/clwang/Data/RJH/RJ_data/SWIM_preprocessed/swim_train.json"


input_data = get_items_from_file(files_list, format='json')
print(input_data[0])
dataset_ =  get_rjh_swim_seg_dataset(
    input_data,
    phase='train',
    spacing=None,
    winlevel=None,
    in_channels=1,
    crop_size=(96,96,64),
    preload=0,
    augment_ratio=1,
    orientation='RAI',
    cache_dir='./',
    verbose=False
)

print("Len:", len(dataset_))
for i, data in enumerate(dataset_):
    print(i, type(data['image']),  data['image'].shape)
    img = data['image'].numpy()
    print('mean:', np.mean(img), 'std:', np.std(img))
    print('0.1 per:', np.percentile(img, 0.1), '99.9 per:', np.percentile(img, 99.9))
    print('0.2 per:', np.percentile(img, 0.2), '99.8 per:', np.percentile(img, 99.8))
   
    save_dir = '/homes/clwang/Data/RJH'
    nib.save( nib.Nifti1Image(data['image'].numpy().squeeze(), np.eye(4)), os.path.join(save_dir, f'{i}-img.nii.gz') )
    # nib.save( nib.Nifti1Image(data[0]['label'].numpy().squeeze().astype(np.uint8), np.eye(4)), os.path.join(save_dir, f'{i}-roi.nii.gz') )
    input('Continue?')