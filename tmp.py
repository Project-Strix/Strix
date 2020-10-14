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
for i in range(10):
    gamma = 0.6 + i*0.2 
    print('Gamma:', gamma)
    deform = Compose([
        ScaleIntensity(),
        AdjustContrast(gamma=gamma)
        ])
    #data = nonlinear_transformation(loader(fname).squeeze())
    data = deform(loader(fname)[np.newaxis,...])
    print('Range:', np.min(data), np.max(data))
    plt.imshow(data.squeeze(), cmap=plt.cm.gray)
    plt.show()

# %%
import os, tqdm, json
from pathlib import Path
import matplotlib.pyplot as plt
from monai.transforms import *


folder = Path(r"\\mega\clwang\Data\object-CXR\train")
files = [ f for f in folder.iterdir()]

# json_file = r"\\mega\clwang\Data\object-CXR\train_data_list_win.json"
# with open(json_file, 'r') as f:
#     files = json.load(f)

loader = Compose([LoadPNGd(keys='image'), AddChanneld(keys='image')]) 
for i, file in enumerate(files):
    # if file.stem not in ['02988','02188','01231','05052']:
    #     continue

    if not os.path.isfile(file):
        continue

    print(file.stem)
    image = loader({'image':file})
    new_image = FixedResized(keys='image', spatial_size=512)(image)
    print('\n', image['image'].shape, new_image['image'].shape)
    plt.imshow(new_image['image'].squeeze(), cmap=plt.cm.gray)
    plt.show()

    # except:
    #     print("Error in read:", file.stem)
    #     continue
    input()
    # ratio = np.count_nonzero(image < 100) / image.size
    # #plt.hist(image.ravel(), bins=128)
    # if ratio > 0.75:
    #     plt.imshow(image, cmap=plt.cm.gray)
    #     plt.show()
    #     print('Remove:', file.stem)
    #     os.remove(file)

# %%
