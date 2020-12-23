import os
import nibabel as nib
import numpy as np
from utils_cw import get_items_from_file
from monai_ex.transforms import *
from monai_ex.data import DataLoader
from medlp.data_io.segmentation_dataset import SegmentationDataset2D
import torchvision
import matplotlib.pyplot as plt

files_list = "/homes/clwang/Data/picc/picc_dcm_nii.json"

def PICC_dcm_seg_dataset(files_list, 
                         phase, 
                         spacing=(0.3,0.3), 
                         winlevel=(421,2515), 
                         in_channels=1, 
                         image_size=(1024,1024),
                         crop_size=None, 
                         preload=1.0, 
                         augment_ratio=0.4):
    assert in_channels == 1, 'Currently only support single channel input'
    
    if phase == 'train':
        additional_transforms = [
            RandRotated(keys=["image","label"], range_x=5, range_y=5, prob=augment_ratio, padding_mode='zeros'),
            RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[1])
        ]
    elif phase == 'valid':
        additional_transforms = []

    dataset = SegmentationDataset2D(
        files_list,
        loader=[LoadImageD(keys='image'), LoadNiftiD(keys='label')],
        channeler=TransposeD(keys='label'), #AsChannelFirstD(keys='label'),
        orienter=None,
        spacer=SpacingD(keys=["image","label"], pixdim=spacing),
        rescaler=ScaleIntensityViaDicomD(keys="image", win_center_key='0028|1050', win_width_key='0028|1051', clip=True),
        resizer=ResizeWithPadOrCropd(keys=["image","label"], spatial_size=image_size),
        cropper=None,
        to_tensor=None,
        additional_transforms=additional_transforms,    
        preload=preload
    ).get_dataset()

    return dataset

dataset_ = PICC_dcm_seg_dataset(get_items_from_file(files_list,'json'), 'train', preload=0, augment_ratio=1)
#dataloader = DataLoader(dataset_, batch_size=1, shuffle=True)

for i, data in enumerate(dataset_):
    print(i, type(data), type(data['image']), type(data['label']))

    # grid_image = torchvision.utils.make_grid(data['image'], nrow=2).permute(1, 2, 0)
    # print(grid_image.shape)
    # plt.imsave('/homes/clwang/Data/picc/dcm_grid.png', grid_image, cmap=plt.cm.gray)


    save_dir = '/homes/clwang/Data/picc'
    nib.save( nib.Nifti1Image(1-data['image'], np.eye(4)), os.path.join(save_dir, f'{i}-image.nii.gz') )
    nib.save( nib.Nifti1Image(data['label'], np.eye(4)), os.path.join(save_dir, f'{i}-label.nii.gz') )
    break