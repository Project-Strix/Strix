import pytest
from types import SimpleNamespace as sn
import torch

from strix.models.cnn.engines.selflearning_engines import SelflearningTrainEngine
from strix.models.cnn.nets.dynunet import DynUNet as UNet
from strix.utilities.enum import Phases
from torch.optim import SGD, lr_scheduler
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
        "criterion": "MSE",
    }
    opts1 = sn(**opts1)

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dim", [2, 3])
    def test_segmentation_train_engine(self, device, dim, tmp_path):
        datasets = DatasetRegistry()
        strix_ds = datasets.get(f"{dim}D", "selflearning", "SyntheticData")
        train_ds = get_torch_datast(
            strix_ds, Phases.TRAIN, {"output_nc": 1, "tensor_dim": f"{dim}D", "framework": "selflearning"}, synthetic_data_num=2
        )
        valid_ds = get_torch_datast(
            strix_ds, Phases.VALID, {"output_nc": 1, "tensor_dim": f"{dim}D", "framework": "selflearning"}, synthetic_data_num=2
        )
        writer = SummaryWriter(log_dir=tmp_path / "tensorboard")
        net = UNet(dim, 1, 1, (3,) * 4, (1,) + (2,) * 3, (1,) + (2,) * 3, last_activation="sigmoid")
        optim = SGD(net.parameters(), 0.1)
        self.opts1.phase = Phases.TRAIN
        if device == "cuda":
            self.opts1.amp = True

        trainer = SelflearningTrainEngine(
            self.opts1,
            DataLoader(train_ds, 0, batch_size=self.opts1.n_batch),
            DataLoader(valid_ds, 0, batch_size=self.opts1.n_batch),
            net.to(device),
            torch.nn.MSELoss(),
            optim,
            lr_scheduler.LambdaLR(optim, lr_lambda=lambda x: 1),
            writer,
            torch.device(device),
            tmp_path,
            None,
        )

        trainer.run()
