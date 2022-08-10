import pytest
from types import SimpleNamespace as sn
import torch

from strix import strix_networks, strix_datasets
from strix.models.cnn.engines.classification_engines import ClassificationTestEngine
from strix.utilities.enum import Phases
from strix.utilities.utils import get_torch_datast
from monai_ex.data import DataLoader

class TestClassification:
    opts1 = {
        "amp": False, "output_nc": 1, "save_n_best": 1, "save_epoch_freq": 1, "nni": False, "n_batch": 1,
        "n_epoch": 1, "early_stop": 0, "n_epoch_len": 1, "visualize": False, "criterion": "Dice"
    }
    opts = sn(**opts1)

    @pytest.mark.parametrize("device", ["cuda:0"])  # "cpu"
    @pytest.mark.parametrize("phase", [Phases.TEST_IN, Phases.TEST_EX])
    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("save_img", [True, False])
    @pytest.mark.parametrize("save_prob", [True, False])
    def test_classification_test_engine(self, device, phase, dim, save_img, save_prob, tmp_path):
        strix_ds = strix_datasets.get(f"{dim}D", 'classification', 'RandomData')
        torch_ds = get_torch_datast(strix_ds, Phases.VALID, {"output_nc": 1, "tensor_dim": f"{dim}D"}, synthetic_data_num=4)

        net = strix_networks[f"{dim}D"]["classification"]["vgg9"](
            spatial_dims=dim,
            in_channels=1,
            out_channels=1,
            act='relu',
            norm='batch',
            n_depth=None,
            n_group=1,
            drop_out=0,
            is_prunable=False,
            pretrained=False,
            pretrained_model_path="",
        )
        torch.save(net.state_dict(), tmp_path / "temp_model.pt")

        self.opts.phase = phase
        self.opts.model_path = [tmp_path / "temp_model.pt"]
        self.opts.out_dir = tmp_path
        self.opts.save_image = save_img
        self.opts.save_prob = save_prob

        if 'cuda' in device:
            self.opts.amp = True

        evaluater = ClassificationTestEngine(
            self.opts,
            DataLoader(torch_ds, 0, batch_size=self.opts.n_batch),
            net.to(device),
            torch.device(device),
            "logger_name",
        )

        evaluater.run()