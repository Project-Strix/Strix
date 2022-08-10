import pytest
from strix.utilities.registry import DatasetRegistry

def test_classification_dataset():
    ds = DatasetRegistry()
    dummy_cls_dataset = lambda file_list, phase, opts: [1, 2, 3]
    ds.register('2D', 'classification', 'dummy', '', '', dummy_cls_dataset)

    assert 'dummy' in ds.list('2D', 'classification')
