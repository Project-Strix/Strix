import os
import numpy as np
from utils_cw.augmentations import Compose
from medlp.utilities.transforms import RandCrop2dByPosNegLabelD, bbox_3D
from utils_cw import get_items_from_file
import nibabel as nib
from monai_ex.data import DataLoader, Dataset
from monai_ex.transforms import *

file = r"/homes/clwang/Data/jsph_lung/YHBLXA_YXJB/datalist-train.json"

data_list = get_items_from_file(file, format='json')

cropper = RandCrop2dByPosNegLabelD(
    keys=['image', 'mask'],
    label_key='mask',
    spatial_size=(40,40),
    crop_mode='parallel',
    z_axis=2,
    pos=1,
    neg=0,
)

trans = Compose(
    [
        LoadImageD(keys=["image", "mask"]),
        AddChannelD(keys=['image', 'mask']),
        cropper,
    ]
)

dataset = Dataset(data_list, transform=trans)
dataloader = DataLoader(dataset, num_workers=0, batch_size=2, shuffle=False)

for i, data in enumerate(dataloader):
    print('Dataloader:', i, data['image'].shape)
    nib.save( nib.Nifti1Image( data['image'][0].numpy(), np.eye(4) ), f'/homes/clwang/Data/{i}crop.nii.gz' )
    nib.save( nib.Nifti1Image( data['mask'][0].numpy(), np.eye(4) ), f'/homes/clwang/Data/{i}crop_m.nii.gz')

# for i, data in enumerate(data_list):
#     image = nib.load(data['image']).get_fdata()[np.newaxis, ...]
#     mask = nib.load(data['mask']).get_fdata()[np.newaxis, ...]

#     print(image.shape, mask.shape)
#     input_data = {'image': image, 'mask':mask}
#     vol = cropper(input_data)
#     print('Crop shape:', vol[0]['image'].shape)