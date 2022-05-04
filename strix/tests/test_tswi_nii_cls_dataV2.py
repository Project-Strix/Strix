import os
from strix.data_io import CLASSIFICATION_DATASETS
from monai_ex.transforms import *
from monai_ex.data import DataLoader

from utils_cw import get_items_from_file, get_connected_comp
import nibabel as nib
import numpy as np


# files_list = "/homes/clwang/Data/RJH/STS_tSWI/datalist_wi_mask@1130_1537-test.json"
# dataset_ = get_rjh_tswi_cls_datasetV2(
#     get_items_from_file(files_list, format='json'),
#     phase='test',
#     spacing=(0.6667,0.6667,1.34),
#     winlevel=None,
#     in_channels=1,
#     crop_size=(32,32,16),
#     preload=0,
#     augment_ratio=1,
#     orientation='RAI',
#     cache_dir='./',
#     verbose=False
# )

files_list = "/homes/clwang/Data/RJH/STS_tSWI/datalist_wi_mask@1229_2131-train.json"

dataset_ = CLASSIFICATION_DATASETS["2D"]["rjh_tswi_oneside"](
    get_items_from_file(files_list),
    phase="train",
    opts={"preload": 0, "experiment_path": "./"},
)
dataloader = DataLoader(dataset_, batch_size=3, shuffle=True)

print("Len:", len(dataset_))
for i, data in enumerate(dataloader):
    print(i, type(data["image"]), data["image"].shape, "label:", data["label"].shape)

    save_dir = "/homes/clwang/Data/RJH"
    nib.save(
        nib.Nifti1Image(data["image"].numpy().squeeze()[0], np.eye(4)), os.path.join(save_dir, f"{i}-test-img.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(data["mask"].numpy().squeeze()[0].astype(np.uint8), np.eye(4)), os.path.join(save_dir, f"{i}-roi.nii.gz"),
    )
    if i > 5:
        break
