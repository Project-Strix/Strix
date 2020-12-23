import os
from medlp.data_io.rjh_dataset import get_rjh_tswi_cls_datasetV2, get_rjh_tswi_cls_dataset2D
#from medlp.data_io.base_dataset.segmentation_dataset import SegmentationDataset3D
from medlp.utilities.transforms import LabelMorphologyD
from medlp.utilities.utils import bbox_3D
from monai_ex.transforms import *
from monai_ex.data import DataLoader

from utils_cw import get_items_from_file, get_connected_comp
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi


files_list = "/homes/clwang/Data/RJH/STS_tSWI/datalist_wi_mask@1130_1537-test.json"

# for data_dict in get_items_from_file(files_list, format='json'):
#     image_nii = nib.load(data_dict['image'])
#     mask_nii = nib.load(data_dict['mask'])
    
#     mask_data = mask_nii.get_fdata().squeeze()
#     label_L = get_connected_comp(mask_data==1, topK=1, min_th=0, verbose=True)
#     label_R = get_connected_comp(mask_data==2, topK=1, min_th=0, verbose=True)
#     x, y, i, j, k, m = bbox_3D(label_L)
#     xx, yy, ii, jj, kk, mm = bbox_3D(label_R)
#     print('Name:', os.path.basename(data_dict['mask']))
#     print('\tRoi size:', (y-x,j-i,m-k), (mm-kk,jj-ii,yy-xx))
#     new_mask_data = np.zeros_like(mask_data)
#     new_mask_data[label_L>0] = 1
#     new_mask_data[label_R>0] = 2 
#     nib.save( nib.Nifti1Image(new_mask_data, mask_nii.affine), data_dict['mask'].replace('_seg','_seg_postprocess') )

# for i, data_dict in enumerate(get_items_from_file(files_list, format='json')):
#     image_nii = nib.load(data_dict['image'])
#     mask_nii = nib.load(data_dict['mask'])

#     mask_data = mask_nii.get_fdata()[np.newaxis,...]
#     print(mask_data.shape)
#     new_mask = Rand3DElastic(sigma_range=(6,10), magnitude_range=(100,150), 
#                              prob=1, mode="nearest", padding_mode='zeros')(mask_data)
#     save_dir = '/homes/clwang/Data/RJH'
#     nib.save( nib.Nifti1Image(mask_data.squeeze(), np.eye(4)), os.path.join(save_dir, f'{i}-mask.nii.gz') )
#     nib.save( nib.Nifti1Image(new_mask.squeeze().astype(np.uint8), np.eye(4)), os.path.join(save_dir, f'{i}-new-mask.nii.gz') )

dataset_ = get_rjh_tswi_cls_datasetV2(
    get_items_from_file(files_list, format='json'),
    phase='test',
    spacing=(0.6667,0.6667,1.34),
    winlevel=None,
    in_channels=1,
    crop_size=(32,32,16),
    preload=0,
    augment_ratio=1,
    orientation='RAI',
    cache_dir='./',
    verbose=False
)
dataloader = DataLoader(dataset_, batch_size=3, shuffle=True)

print("Len:", len(dataset_))
for i, data in enumerate(dataloader):
    print(i, type(data['image']),  data['image'].shape, 'label:', data['label'].shape)
   
    save_dir = '/homes/clwang/Data/RJH'
    nib.save( nib.Nifti1Image(data['image'].numpy().squeeze()[0], np.eye(4)), os.path.join(save_dir, f'{i}-test-img.nii.gz') )
    nib.save( nib.Nifti1Image(data['mask'].numpy().squeeze()[0].astype(np.uint8), np.eye(4)), os.path.join(save_dir, f'{i}-roi.nii.gz') )
    if i > 5:
        break


