
#%%
import os
from strix.data_io.rjh_dataset import get_rjh_tswi_seg_dataset
from strix.utilities.transforms import MarginalCropByMaskD, bbox_3D

from utils_cw import get_items_from_file
import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

file = "/homes/clwang/Data/RJH/STS_tSWI/datalist_wi_mask@1124_1034.json"
#file = "/homes/clwang/Data/strix_exp/segmentation/rjh_tswi/res-unet-96,96,64-CE-DCE-BN-adamw-plateau-1123_1609-Dil-Elastic-3cls/train_files"
data_list = get_items_from_file(file, format='json')

# sizes = np.zeros([len(data_list), 3])
# for i, data in enumerate(data_list):
#     mask = nib.load(data['label']).get_fdata().squeeze()
#     ccs, num = ndi.label(mask)
#     for k in [1,2]:
#         x1, x2, y1, y2, z1, z2 = bbox_3D(ccs == k)
#         sizes[i] = np.array([x2-x1, y2-y1, z2-z1])

# print(sizes)
# print('mean:', np.mean(sizes, axis=0))
# print('max:', np.max(sizes, axis=0))
# print('min:', np.min(sizes, axis=0))


cropper = MarginalCropByMaskD(keys='image', mask_key='mask', label_key='label', margin_size=(5,5,2), divide_by_k=16, keep_largest=True)
for i, data in enumerate(data_list):
    image = nib.load(data['image']).get_fdata().transpose(3,0,1,2)
    mask = nib.load(data['mask']).get_fdata().transpose(3,0,1,2)

    input_data = {'image': image, 'label':data['label'], 'mask':mask}
    vol = cropper(input_data)
    print('Shape:', vol['image'].shape, 'Label:', vol['label'])

    #nib.save( nib.Nifti1Image(vol['image'].squeeze(), np.eye(4)), os.path.join("/homes/clwang/Data/RJH", os.path.basename(data['image'])))

