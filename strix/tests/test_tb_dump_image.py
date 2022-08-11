import pytest
from types import SimpleNamespace as sn
import torch

from strix import strix_datasets
from strix.datasets.synthetic_dataset import GenerateSyntheticDataD
from strix.models.cnn.engines.segmentation_engines import SegmentationTrainEngine
from strix.models.cnn.nets.dynunet import DynUNet as UNet
from strix.utilities.enum import Phases
from torch.optim import SGD, lr_scheduler
from strix.models.cnn.losses import DiceLoss
from strix.utilities.utils import get_torch_datast
from monai_ex.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

opts1 = {
    "amp": False,
    "output_nc": 1,
    "save_n_best": 1,
    "save_epoch_freq": 1,
    "nni": False,
    "n_batch": 1,
    "n_epoch": 2,
    "early_stop": 0,
    "n_epoch_len": 1,
    "visualize": False,
    "criterion": "Dice",
}
opts1 = sn(**opts1)


@pytest.mark.parametrize("interval", [1, 5])
def test_segmentation_train_engine(interval, tmp_path):
    dim = 2
    strix_dataset = strix_datasets.get(f"{dim}D", "segmentation", "SyntheticData")
    if strix_dataset is None:
        raise ValueError("strix dataset is None")

    opts = {"output_nc": 1, "tensor_dim": f"{dim}D", "framework": "segmentation"}
    train_torch_ds = get_torch_datast(strix_dataset, Phases.TRAIN, opts, 3)
    valid_torch_ds = get_torch_datast(strix_dataset, Phases.VALID, opts, 3)
    if train_torch_ds is None or valid_torch_ds is None:
        raise ValueError("Torch dataset generate failed!")

    writer = SummaryWriter(log_dir=tmp_path / "tensorboard")
    net = UNet(dim, 1, 1, (3,) * 4, (1,) + (2,) * 3, (1,) + (2,) * 3)
    optim = SGD(net.parameters(), 0.1)
    device = "cuda:0"
    opts1.phase = Phases.TRAIN
    opts1.amp = True
    opts1.tb_dump_img_interval = interval

    trainer = SegmentationTrainEngine(
        opts1,
        DataLoader(train_torch_ds, 0, batch_size=opts1.n_batch),
        DataLoader(valid_torch_ds, 0, batch_size=opts1.n_batch),
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

    logdir = tmp_path / "tensorboard"
    logfile = list(filter(lambda x: "events" in str(x), logdir.iterdir()))[0]
    print("interval:", interval, "logfile size:", logfile.stat().st_size)
