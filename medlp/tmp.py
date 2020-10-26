# %%
import os, torch
from monai.networks.nets import UNet, DynUNet, DenseNet

pretrain_state_dict = torch.load("/homes/clwang/Data/picc/exp/selflearning/Obj_CXR/unet-512,512-MSE-sgd-const-1015_1215-crop/Models/checkpoint_epoch=1000.pt")
mod = DynUNet(spatial_dims=2,
              in_channels=1,
              out_channels=2,
              norm_name="batch",
              kernel_size=(3, 3, 3, 3, 3, 3),
              strides=(1, 2, 2, 2, 2, 2),
              #upsample_kernel_size=(3, 3, 3, 3, 3, 3),
              deep_supervision=False,
              deep_supr_num=1,
              res_block=True,
              last_activation=None)

model_dict = mod.state_dict()
filtered_dict = {k: v for k, v in pretrain_state_dict['net'].items() if v.shape == model_dict[k].shape}
model_dict.update(filtered_dict)
mod.load_state_dict(model_dict)

os.sys.exit()
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
import os
import numpy as np 
import matplotlib.pyplot as plt
from monai.transforms import *

params = {
    'prob':1.0,
    'spacing':(500, 300),
    'magnitude_range':(5, 15),
    'scale_range':(0, 0),
    'translate_range':(0, 0),
    'padding_mode':"border",
}
augment_ratio = 0.6
fname = r"\\alg-cloud2\Incoming\YYQ\object-CXR\raw-data\data-jpg\dev\08001.jpg"
tsf = Compose([
    LoadPNGd(keys='image'),
    AddChanneld(keys='image'),
    adaptor(RandLocalPixelShuffle(prob=augment_ratio, num_block_range=[50,200]), "image"),
    Rand2DElasticd(keys="image", prob=augment_ratio, spacing=(300, 300), magnitude_range=(10, 20), padding_mode="border"),
    RandomSelect([
        adaptor(RandImageInpainting(prob=1, num_block_range=(3,5)), 'image'),
        adaptor(RandImageOutpainting(prob=1, num_block_range=(5,8)), 'image'),
    ], prob=0.8),
])

for _ in range(5):
    data = tsf({'image':fname})
    print(data['image'].shape)
    plt.imshow(data['image'].squeeze(), cmap=plt.cm.gray)
    plt.show()

# %%
import random
import matplotlib.pyplot as plt
from monai.transforms import *

deform = Rand2DElastic(
    prob=1.0,
    magnitude_range=(10, 20),
    spacing=(300, 300),
    # scale_range=(0, 0),
    # translate_range=(0, 0),
    padding_mode="border",
)

loader = LoadPNG(image_only=True)

fname = r"\\alg-cloud2\Incoming\YYQ\object-CXR\raw-data\data-jpg\dev\08001.jpg"
for i in range(1):
    gamma = 0.6 + i*0.2 
    print('Gamma:', gamma)
    deform = Compose([
        ScaleIntensity(),
        AdjustContrast(gamma=gamma)
        ])
    #data = nonlinear_transformation(loader(fname).squeeze())
    data = deform(loader(fname)[np.newaxis,...])
    print('Range:', np.min(data), np.max(data))
    plt.imshow(data.squeeze(), cmap=plt.cm.gray, vmin=0.2, vmax=1.8)
    plt.show()

# %%
import os, tqdm, json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from monai.transforms import *
from scipy import ndimage

folder = Path(r"\\mega\clwang\Data\object-CXR\train")
files = [ f for f in folder.iterdir()]

image_size = 1024
augment_ratio = 0.5
transforms = Compose([
    LoadPNG(image_only=True, grayscale=True),
    AddChannel(),
    ScaleIntensity(),
    FixedResize(spatial_size=image_size, mode=InterpolateMode.AREA),
    #RandSpatialCrop(roi_size=384, random_size=False),
    #resizer,
    #RandAdjustContrast(prob=augment_ratio, gamma=(0.5, 1.5)),
    RandFlip(prob=augment_ratio, spatial_axis=[1]),
    RandNonlinear(prob=augment_ratio),
    RandLocalPixelShuffle(prob=augment_ratio, num_block_range=[10000,10001]),
    RandomSelect([
        RandImageInpainting(prob=1, num_block_range=(3,5)),
        RandImageOutpainting(prob=1, num_block_range=(6,9)),
    ], prob=augment_ratio),
    CastToType(dtype=np.float32),
])

loader = Compose([LoadPNGd(keys='image', grayscale=True), 
                  ScaleIntensityd(keys='image'), 
                  AddChanneld(keys='image'), 
                  RandSpatialCropd(keys='image', roi_size=512, random_size=False),
                  adaptor(RandLocalPixelShuffle(prob=augment_ratio, num_block_range=[100,1000]), "image")])
for i, file in enumerate(files):
    if i > 10:
        break

    if not os.path.isfile(file):
        continue

    print(file.stem)    
    x = loader(file)
    #x = transforms(file)

    #new_image = FixedResized(keys='image', spatial_size=512)(image)
    plt.subplot(1,2,1)
    plt.imshow(x.squeeze(), cmap=plt.cm.gray)
    plt.subplot(1,2,2)
    plt.imshow(y.squeeze(), cmap=plt.cm.gray)
    plt.show()
    input()



# %%
import os, tqdm, json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from monai.transforms import *
from scipy import ndimage

transforms = Compose([
    LoadHdf5d(keys=["image","label"], h5_keys=["data","label"], dtype=[np.float32, np.int64]),
    AddChanneld(keys=["image", "label"]),
    RandAdjustContrastd(keys=["image","label"], prob=augment_ratio, gamma=(0.9,1.1)),
    RandSpatialCropd(keys=["image", "label"], roi_size=crop_size, random_size=False),
    RandGaussianNoised(keys="image", prob=augment_ratio, std=0.2),
    RandRotated(keys=["image","label"], range_x=10, range_y=10, range_z=5, prob=augment_ratio),
    RandFlipd(keys=["image","label"], prob=augment_ratio, spatial_axis=[0]),
    CastToTyped(keys=["image","label"], dtype=[np.float32, np.int64]),
    ToTensord(keys=["image", "label"])
])

# %%
