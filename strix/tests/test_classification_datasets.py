import pytest
from strix.data_io import CLASSIFICATION_DATASETS

def test_classification_dataset():
    dummy_cls_dataset = lambda file_list, phase, opts: [1,2,3]
    CLASSIFICATION_DATASETS.register('2D', 'dummy', '', '', dummy_cls_dataset)
    
    assert 'dummy' in CLASSIFICATION_DATASETS['2D'].keys()
