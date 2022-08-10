import pytest
from monai.data import Dataset
from strix.data_io import StrixDataset

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
    }
]


def test_basic_clf_dataset():
    data = StrixDataset(
        filelist=filelist,
        loader=lambda x: x,
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
        check_data=True,
        verbose=False
    )

    assert len(data) == 2
    assert list(data[0].keys()) == ['image', 'mask', 'label']