#%%
import os
import numpy as np 
from monai.transforms import *

folder = Path(r"\\mega\clwang\Data\picc\exp\segmentation\rib-unet-0,0-CE-adam-step-1001_0014-deeper\Test")
out_dir = folder.parent.joinpath('Test1')
picc_foler = Path(r"\\mega\clwang\Data\picc\exp\segmentation\picc_h5-scnn-1024,1024-WCE-sgd-const-0917_1939-redo3-amp\Test_old")
files = list(folder.glob('*/*.nii.gz'))

zoomer = Zoom(zoom=2, mode=InterpolateMode.NEAREST, keep_size=False)
spacer = Spacing(pixdim=[0.3,0.3])
for f in files:
    nii = nib.load(f)
    data, affine = nii.get_fdata().squeeze()[...,1], nii.affine

    new_data = zoomer(data[np.newaxis,...])
    new_data = spacer(new_data, affine=affine)

    print( f.stem, new_data[0].shape)
    casename = f.stem.replace('_seg.nii','')
    picc_file = str(picc_foler/casename/casename)+'_seg.nii.gz'
    if not os.path.isfile(picc_file):
        print('Cannot find', picc_file)
        continue
    picc_nii = nib.load(picc_file)
    picc_shape = picc_nii.shape
    output = ResizeWithPadOrCrop(spatial_size=picc_shape[:-2])(new_data[0])
    print('Output shape:', output.shape)
    os.makedirs(out_dir, exist_ok=True)
    nib.save( nib.Nifti1Image(output.squeeze(), picc_nii.affine), out_dir.joinpath(f.stem))

# %%

