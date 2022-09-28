import pytest
from strix.utilities.enum import FRAMEWORKS
from strix import strix_losses, strix_datasets


@pytest.mark.parametrize("framework", ["classification", "seg"])
def test_loss_registry(framework):
    loss_name = "dummy_loss"

    if framework in FRAMEWORKS:
        strix_losses.register(framework, loss_name, lambda x: x)
        assert strix_losses.get(framework, loss_name) is not None
    else:
        with pytest.raises(AssertionError):
            strix_losses.register(framework, loss_name, lambda x: x)


@pytest.mark.parametrize("dim", ["2D", 3])
@pytest.mark.parametrize("framework", ["classification", "seg"])
def test_dataset_registry(dim, framework):
    ds_name = "dummy_dataset"

    if framework in FRAMEWORKS:
        strix_datasets.register(dim, framework, ds_name, "train_files", None, lambda x: x)
        assert strix_datasets.get(dim, framework, ds_name) is not None
    else:
        with pytest.raises(AssertionError):
            strix_datasets.register(dim, framework, ds_name, "train_files", None)
