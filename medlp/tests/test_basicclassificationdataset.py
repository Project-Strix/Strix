from monai.data import Dataset
from strix.data_io import BasicClassificationDataset

filelist = [
    {
        "image": "/homes/clwang/Data/kits19_seg/data/case_00000/segmentation.nii.gz",
        "mask": "/homes/clwang/Data/kits19_seg/data/case_00000/segmentation.nii.gz",
        "label": 1,
    },
    {
        "image": "/homes/clwang/Data/kits19_seg/data/case_00000/segmentation.nii.gz",
        "mask": "/homes/clwang/Data/kits19_seg/data/case_00000/segmentation.nii.gz",
        "label": 1,
    },
    {
        "image": "/homes/clwang/Data/kits19_seg/data/case_00000/segmentation.nii.gz",
        "mask": "/homes/clwang/Data/kits19_seg/data/case_00000/segmentation.nii.gz",
        "label": 1,
    }
]

data = BasicClassificationDataset(
    files_list=filelist,
    loader=None,
    channeler=None,
    orienter=None,
    spacer=None,
    rescaler=None,
    resizer=None,
    cropper=None,
    caster=None,
    to_tensor=None,
    is_supervised=True,
    dataset_type=Dataset,
    dataset_kwargs={},
    verbose=True
)

print(data.input_data)