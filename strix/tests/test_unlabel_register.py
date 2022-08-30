import pytest
from strix.utilities.registry import DatasetRegistry
from strix.utilities.enum import Phases

def test_unlabel_register_error():
    data_regi = DatasetRegistry()

    with pytest.raises(AssertionError):
        @data_regi.register_unlabel("3D", "test_unlabel_data")
        def test_unlabel_fn(filelist, phase, opts):
            pass
    
def test_unlabel_register():
    data_regi = DatasetRegistry()

    @data_regi.register("3D", "test_unlabel_data", "./test_filelist.json")
    def test_label_fn(filelist, phase, opts):
        return

    @data_regi.register_unlabel("3D", "test_unlabel_data")
    def test_unlabel_fn(filelist, phase, opts):
        return True

    assert data_regi["3D"]["test_unlabel_data"]["UNLABEL_FN"]([], 'train', {})
    

def test_unlabel_dataloader():
    from strix.data_io import SEGMENTATION_DATASETS
    from strix.data_io.dataio import get_dataloader
    from types import SimpleNamespace as SN
    from monai_ex.utils.exceptions import DatasetException

    @SEGMENTATION_DATASETS.register("3D", "test_unlabel_data", "./test_filelist.json")
    def test_label_fn(filelist, phase, opts):
        return

    args = {
        "n_batch":1,
        "n_batch_valid":1,
        "n_worker":0,
        "framework": "segmentation",
        "tensor_dim": "3D",
        "data_list": "test_unlabel_data",
        "imbalance_sample": False
    }
    with pytest.raises(DatasetException):
        get_dataloader.__wrapped__(SN(**args), [], Phases.TRAIN, is_unlabel=True)

    @SEGMENTATION_DATASETS.register_unlabel("3D", "test_unlabel_data")
    def test_unlabel_fn(filelist, phase, opts):
        return filelist
    
    get_dataloader.__wrapped__(SN(**args), ['file1', 'file2'], Phases.TRAIN, is_unlabel=True)