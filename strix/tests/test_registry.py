import pytest
from strix.utilities.enum import FRAMEWORKS
from strix.utilities.registry import LossRegistry, DatasetRegistry


@pytest.mark.parametrize("framework", ["classification", "seg"])
def test_loss_registry(framework):
    loss_name = "dummy_loss"
    loss_reg = LossRegistry()

    if framework in FRAMEWORKS:
        loss_reg.register(framework, loss_name, lambda x: x)
        assert loss_reg.get(framework, loss_name) is not None
    else:
        with pytest.raises(AssertionError):
            loss_reg.register(framework, loss_name, lambda x: x)


@pytest.mark.parametrize("dim", ["2D", 3])
@pytest.mark.parametrize("framework", ["classification", "seg"])
def test_dataset_registry(dim, framework):
    ds_name = "dummy_dataset"
    dataset_reg = DatasetRegistry()

    if framework in FRAMEWORKS:
        dataset_reg.register(dim, framework, ds_name, "train_files", None, lambda x: x)
        assert dataset_reg.get(dim, framework, ds_name) is not None
    else:
        with pytest.raises(AssertionError):
            dataset_reg.register(dim, framework, ds_name, "train_files", None)
