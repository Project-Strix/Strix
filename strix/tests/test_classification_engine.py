import pytest
from types import SimpleNamespace as sn
import torch

from strix.models.cnn.engines.classification_engines import ClassificationTestEngine
from strix.models.cnn.nets.dynunet import DynUNet as UNet
from strix.utilities.registry import NetworkRegistry
from strix.utilities.enum import Phases
from torch.optim import SGD, lr_scheduler
from strix.models.cnn.losses import DiceLoss
from strix.data_io import CLASSIFICATION_DATASETS
from monai_ex.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class TestClassification:
    opts1 = {
        "amp": False, "output_nc": 1, "save_n_best": 1, "save_epoch_freq": 1, "nni": False, "n_batch": 1,
        "n_epoch": 1, "early_stop": 0, "n_epoch_len": 1, "visualize": False, "criterion": "Dice"
    }
    opts = sn(**opts1)

    filst_list = [{"image": '1.nii', "label": '1.1.nii'}, {"image": '2.nii', "label": '2.1.nii'},]

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("phase", [Phases.TEST_IN, Phases.TEST_EX])
    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("save_img", [True, False])
    @pytest.mark.parametrize("save_prob", [True, False])
    def test_segmentatation_test_engine(self, device, phase, dim, save_img, save_prob, tmp_path):
        dataset_fn = CLASSIFICATION_DATASETS[f"{dim}D"]["RandomData"]["FN"]
        Networks = NetworkRegistry()
        net = Networks[f"{dim}D"]["classificaton"]["vgg9"](
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
        torch.save(net.state_dict(), tmp_path/"temp_model.pt")

        self.opts.phase = phase
        self.opts.model_path = [tmp_path/"temp_model.pt"]
        self.opts.out_dir = tmp_path
        self.opts.save_image = save_img
        self.opts.save_prob = save_prob
        
        if 'cuda' in device:
            self.opts.amp = True
        
        evaluater = ClassificationTestEngine(
            self.opts,
            DataLoader(dataset_fn(self.filst_list, Phases.VALID, {"output_nc": 1, "tensor_dim": f"{dim}D"}), 0, batch_size=self.opts.n_batch),
            net.to(device),
            torch.device(device),
            "logger_name",
        )

        evaluater.run()