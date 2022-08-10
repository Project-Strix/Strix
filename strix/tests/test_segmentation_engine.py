import pytest
from types import SimpleNamespace as sn
import torch

from strix.models.cnn.engines.segmentation_engines import SegmentationTrainEngine, SegmentationTestEngine
from strix.models.cnn.nets.dynunet import DynUNet as UNet
from strix.utilities.enum import Phases
from torch.optim import SGD, lr_scheduler
from strix.models.cnn.losses import DiceLoss
from strix.utilities.registry import DatasetRegistry
from strix.utilities.utils import get_torch_datast
from monai_ex.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class TestSegmentation:
    opts1 = {
        "amp": False,
        "output_nc": 1,
        "save_n_best": 1,
        "save_epoch_freq": 1,
        "nni": False,
        "n_batch": 1,
        "n_epoch": 1,
        "early_stop": 0,
        "n_epoch_len": 1,
        "visualize": False,
        "criterion": "Dice",
    }
    opts1 = sn(**opts1)


    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dim", [2, 3])
    def test_segmentation_train_engine(self, device, dim, tmp_path):
        datasets = DatasetRegistry()
        strix_dataset = datasets.get(f"{dim}D", "segmentation", "SyntheticData")
        opts = {"output_nc": 1, "tensor_dim": f"{dim}D", "framework": "segmentation"}
        train_torch_ds = get_torch_datast(strix_dataset, Phases.TRAIN, opts, 2)
        valid_torch_ds = get_torch_datast(strix_dataset, Phases.VALID, opts, 2)
        if train_torch_ds is None or valid_torch_ds is None:
            raise ValueError("Torch dataset generate failed!")

        writer = SummaryWriter(log_dir=tmp_path / "tensorboard")
        net = UNet(dim, 1, 1, (3,) * 4, (1,) + (2,) * 3, (1,) + (2,) * 3)
        optim = SGD(net.parameters(), 0.1)
        self.opts1.phase = Phases.TRAIN
        if device == "cuda":
            self.opts1.amp = True

        trainer = SegmentationTrainEngine(
            self.opts1,
            DataLoader(train_torch_ds, 0, batch_size=self.opts1.n_batch),
            DataLoader(valid_torch_ds, 0, batch_size=self.opts1.n_batch),
            net.to(device),
            DiceLoss(include_background=True),
            optim,
            lr_scheduler.LambdaLR(optim, lr_lambda=lambda x: 1),
            writer,
            torch.device(device),
            tmp_path,
            None,
        )

        trainer.run()

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dim", [2, 3])
    @pytest.mark.parametrize("save_img", [True, False])
    @pytest.mark.parametrize("save_prob", [True, False])
    @pytest.mark.parametrize("save_label", [True, False])
    @pytest.mark.parametrize("resample", [True, False])
    def test_segmentatation_test_engine(self, device, dim, save_img, save_prob, save_label, resample, tmp_path):
        datasets = DatasetRegistry()
        strix_dataset = datasets.get(f"{dim}D", "segmentation", "SyntheticData")
        opts = {"output_nc": 1, "tensor_dim": f"{dim}D", "framework": "segmentation"}
        valid_torch_ds = get_torch_datast(strix_dataset, Phases.VALID, opts, 2)
        if valid_torch_ds is None:
            raise ValueError("Torch dataset generate failed!")

        net = UNet(dim, 1, 1, (3,) * 4, (1,) + (2,) * 3, (1,) + (2,) * 3)
        torch.save(net.state_dict(), tmp_path / "temp_model.pt")

        self.opts1.phase = Phases.TEST_IN
        self.opts1.model_path = [tmp_path / "temp_model.pt"]
        self.opts1.out_dir = tmp_path
        self.opts1.save_image = save_img
        self.opts1.resample = resample
        self.opts1.save_prob = save_prob
        self.opts1.save_label = save_label

        if "cuda" in device:
            self.opts1.amp = True

        evaluater = SegmentationTestEngine(
            self.opts1,
            DataLoader(valid_torch_ds, 0, batch_size=self.opts1.n_batch),
            net.to(device),
            torch.device(device),
            "logger_name",
        )

        evaluater.run()
