import pytest
from types import SimpleNamespace as sn
import torch

from strix.models.cnn.engines.selflearning_engines import SelflearningTrainEngine
from strix.models.cnn.nets.dynunet import DynUNet as UNet
from strix.utilities.enum import Phases
from torch.optim import SGD, lr_scheduler
from strix.models.cnn.losses import DiceLoss
from strix.data_io import SELFLEARNING_DATASETS 
from monai_ex.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

class TestSegmentation:
    opts1 = {
        "amp": False, "output_nc": 1, "save_n_best": 1, "save_epoch_freq": 1, "nni": False, "n_batch": 1,
        "n_epoch": 1, "early_stop": 0, "n_epoch_len": 1, "visualize": False, "criterion": "MSE"
    }
    opts1 = sn(**opts1)

    filst_list = [{"image": '1.nii', "label": '1.1.nii'}, {"image": '2.nii', "label": '2.1.nii'}]

    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    @pytest.mark.parametrize("dim", [2, 3])
    def test_segmentation_train_engine(self, device, dim, tmp_path):
        dataset_fn = SELFLEARNING_DATASETS[f"{dim}D"]["SyntheticData"]["FN"]
        writer = SummaryWriter(log_dir=tmp_path/"tensorboard")
        net = UNet(dim, 1, 1, (3,)*4, (1,)+(2,)*3, (1,)+(2,)*3, last_activation='sigmoid')
        optim = SGD(net.parameters(), 0.1)
        self.opts1.phase = Phases.TRAIN
        if device == 'cuda':
            self.opts1.amp = True

        trainer = SelflearningTrainEngine(
            self.opts1,
            DataLoader(dataset_fn(self.filst_list, Phases.TRAIN, {"output_nc":1, "tensor_dim": f"{dim}D", "framework": "selflearning"}), 0, batch_size=self.opts1.n_batch),
            DataLoader(dataset_fn(self.filst_list, Phases.VALID, {"output_nc":1, "tensor_dim": f"{dim}D", "framework": "selflearning"}), 0, batch_size=self.opts1.n_batch),
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