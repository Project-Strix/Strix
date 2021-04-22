
#%%import nibabel as nib
import os
from pathlib import Path
import nibabel as nib
import numpy as np

from medlp.utilities.transforms import SeparateCropSTSdataD, ExtractSTSlicesD
from monai_ex.data import DataLoader, Dataset
from monai_ex.transforms import (
    Compose,
    AddChannelD,
    AsChannelFirstD,
    MapTransform,
    SqueezeDimD
)

#%%
folder = Path(
    # "/homes/clwang/Data/medlp_exp/segmentation/rjh_tswi_v2/res-unet-CE-DCE-BN-radam-plateau-1229_2131-CV5/0-th/Test@0104_1213"
    "/homes/yliu/Data/clwang_data/rjh_tswi_v2"
)

fnames = [
      {"image": str(folder / "000288/000288_image.nii.gz"), "mask": str(folder / "000288/000288_seg.nii.gz"), "label": [1, 0]},
      #{"image": str(folder / "000048/000048_image.nii.gz"), "mask": str(folder / "000048/000048_seg.nii.gz"), "label": [1, 1]},
]

fnames = []
for img in folder.rglob('*_image.nii.gz'):
    seg = str(img).replace('_image', '_seg')
    item = {
        'image': str(img),
        'mask': str(seg),
        'label': [0, 0]
    }
    fnames.append(item)

class TestD(MapTransform):
    def __init__(self, keys):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        print(type(d['image']))
        print(d['image'].shape)
        return d

trans = Compose(
    [
        LoadImageD(keys=["image", "mask"]),
        AddChannelD(keys=['image', 'mask']),
        SeparateCropSTSdataD(
            keys=["image", "mask"],
            mask_key="mask",
            label_key='label',
            crop_size=(32, 32, 16),
            # margin_size=(3, 3, 2),
            labels=[1, 2],
            flip_label=2,
            flip_axis=1,
        ),
        ExtractSTSlicesD(keys=['image', 'mask'], mask_key='mask'),
        # ToTensorD(keys=['image', 'mask']),
        SqueezeDimD(
            keys=['image', 'mask'],
            dim=0
        ),
        AsChannelFirstD(
            keys=['image', 'mask']
        )
    ]
)

dataset = Dataset(fnames, transform=trans)
dataloader = DataLoader(dataset, num_workers=0, batch_size=1, shuffle=False)

for i, data in enumerate(dataloader):
    print('Dataloader:', data['label'])

    nib.save(nib.Nifti1Image(data['image'][0].numpy().squeeze(), np.eye(4)), f'/homes/clwang/crop0_{i}.nii.gz')
    #nib.save(nib.Nifti1Image(data['mask'][0].numpy().squeeze(), np.eye(4)), f'/homes/clwang/mask0_{i}.nii.gz')
    nib.save(nib.Nifti1Image(data['image'][1].numpy().squeeze(), np.eye(4)), f'/homes/clwang/crop1_{i}.nii.gz')
    #nib.save(nib.Nifti1Image(data['mask'][1].numpy().squeeze(), np.eye(4)), f'/homes/clwang/mask1_{i}.nii.gz')
    if i > 3:
        os.sys.exit()


#%%
import numpy as np

from monai.data import DataLoader, Dataset

dataset = Dataset(np.arange(10), transform=lambda x: [x, x*2])
dataloader = DataLoader(dataset, num_workers=0, batch_size=3)
for dat in dataloader:
    print(dat)
# %%
