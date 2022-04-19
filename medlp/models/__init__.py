from pathlib import Path
from utils_cw import check_dir
import torch
from medlp.utilities.registry import NetworkRegistry

CLASSIFICATION_ARCHI = NetworkRegistry()
SEGMENTATION_ARCHI = NetworkRegistry()
SELFLEARNING_ARCHI = NetworkRegistry()
MULTITASK_ARCHI = NetworkRegistry()
SIAMESE_ARCHI = NetworkRegistry()

ARCHI_MAPPING = {
    "segmentation": SEGMENTATION_ARCHI,
    "classification": CLASSIFICATION_ARCHI,
    "selflearning": SELFLEARNING_ARCHI,
    "multitask": MULTITASK_ARCHI,
    "siamese": SIAMESE_ARCHI,
}

from medlp.models.cnn.utils import print_network, PolynomialLRDecay
from medlp.models.cnn.layers.radam import RAdam
from medlp.models.cnn.layers.ranger21 import Ranger21
from medlp.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from medlp.data_io import DATASET_MAPPING
from medlp.utilities.utils import get_attr_
from medlp.models.cnn.losses import LOSS_MAPPING, ContrastiveLoss
from medlp.utilities.imports import import_file
from medlp.utilities.enum import Frameworks
from medlp.configures import config as cfg
from medlp.models.cnn.cnn_nets import *
from medlp.models.transformer.transformer_nets import *


external_dataset_dir = Path(cfg.get_medlp_cfg("EXTERNAL_NETWORK_DIR"))
if external_dataset_dir.is_dir():
    for f in external_dataset_dir.glob("*.py"):
        import_file(f.stem, str(f))


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


def get_loss_fn(framework: str, loss_name: str, loss_params: dict, output_nc: int, deep_supervision: bool = False):
    loss_type = LOSS_MAPPING[framework][loss_name]

    if output_nc == 1:
        kwargs = {
            "include_background": True,
            "sigmoid": True,
            "softmax": False,
            "to_onehot_y": False,
        }
    else:
        kwargs = {
            "include_background": False,
            "sigmoid": False,
            "softmax": True,
            "to_onehot_y": True,
        }

    if "Dice" in loss_type.__name__:
        kwargs.update(loss_params)
        loss = loss_type(**kwargs)
    else:
        loss = loss_type(**loss_params)

    if deep_supervision:
        raise NotImplementedError
        loss = DeepSupervisionLoss(loss)

    return loss


def get_network(opts):
    assert (
        hasattr(opts, "model_name")
        and hasattr(opts, "input_nc")
        and hasattr(opts, "tensor_dim")
        and hasattr(opts, "output_nc")
    )
    options = vars(opts).copy()
    model_name = options.get("model_name")
    dim = 2 if options.pop("tensor_dim") == "2D" else 3
    in_channels = options.pop("input_nc")
    out_channels = options.pop("output_nc")

    n_depth = options.pop("n_depth", -1)
    pretrained = options.pop("pretrained", False)
    act = options.pop("layer_act", "relu")
    norm = options.pop("layer_norm", "batch")
    is_prunable = options.pop("snip", False)
    # crop_size = get_attr_(opts, 'crop_size', None)
    # bottleneck_size = get_attr_(opts, 'bottleneck_size', 7)
    drop_out = options.pop("dropout", None)
    n_group = options.pop("n_group", 1)  # used for multi-group archi
    pretrained_model_path = options.pop("pretrained_model_path", None)

    siamese = None
    siamese_latent_dim = options.pop("latent_dim", 512)
    if ARCHI_MAPPING[opts.framework] == SIAMESE_ARCHI:
        loss_type = LOSS_MAPPING[opts.framework][opts.criterion]
        siamese = "single" if loss_type == ContrastiveLoss else "multi"
        raise NotImplementedError

    try:
        model = ARCHI_MAPPING[opts.framework][opts.tensor_dim][opts.model_name]
    except:
        raise ValueError(f"Cannot find registered model: {opts.model_name}")
    else:
        return model(
            dim,
            in_channels,
            out_channels,
            act,
            norm,
            n_depth,
            n_group,
            drop_out,
            is_prunable,
            pretrained,
            pretrained_model_path,
            **options,  # Todo: Only pass network-related kwargs instead of all
        )


def get_engine(opts, train_loader, test_loader, writer=None):
    """Generate engines for specified config.

    Args:
        opts (SimpleNamespace): All arguments from cmd line.
        train_loader (DataLoader): Pytorch dataload for training dataset.
        test_loader (DataLoader): Pytorch dataload for validation dataset.
        writer (SummaryWriter, optional): Tensorboard SummaryWriter. Defaults to None.

    Raises:
        NotImplementedError: Raise error if using undefined loss function.
        NotImplementedError: Raise error if using undefined optim function.

    Returns:
        list: Return engine, net, loss
    """
    # Print the model type
    print("\nInitialising model {}".format(opts.model_name))
    weight_decay = get_attr_(opts, "l2_weight_decay", 0.0)
    nesterov = get_attr_(opts, "nesterov", False)
    momentum = get_attr_(opts, "momentum", 0.0)
    valid_interval = get_attr_(opts, "valid_interval", 5)

    frame, dim, data = opts.framework, opts.tensor_dim, opts.data_list
    multi_input_keys = DATASET_MAPPING[frame][dim][data].get("M_IN", None)
    multi_output_keys = DATASET_MAPPING[frame][dim][data].get("M_OUT", None)

    device = torch.device("cuda") if opts.gpus != "-1" else torch.device("cpu")
    model_dir = check_dir(opts.experiment_path, "Models")

    loss = lr_scheduler = None
    if opts.framework == Frameworks.MULTITASK.value:
        subloss1 = get_loss_fn(opts.subtask1, opts.criterion[0], opts.loss_params_task1, opts.output_nc[0], opts.deep_supervision)
        subloss2 = get_loss_fn(opts.subtask2, opts.criterion[1], opts.loss_params_task2, opts.output_nc[1], opts.deep_supervision)
        loss = LOSS_MAPPING[opts.framework]["CombinationLoss"](subloss1, subloss2, aggregate="sum")
    else:
        loss = get_loss_fn(opts.framework, opts.criterion, opts.loss_params, opts.output_nc, opts.deep_supervision)

    net_ = get_network(opts)

    if len(opts.gpu_ids) > 1:  # and not opts.amp:
        net = torch.nn.DataParallel(net_.to(device))
    else:
        net = net_.to(device)

    if opts.visualize:
        print_network(net)

    if opts.optim == "adam":
        optim = torch.optim.Adam(net.parameters(), opts.lr, weight_decay=weight_decay)
    elif opts.optim == "sgd":
        optim = torch.optim.SGD(
            net.parameters(),
            opts.lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
        )
    elif opts.optim == "adamw":
        optim = torch.optim.AdamW(net.parameters(), opts.lr, weight_decay=weight_decay)
    elif opts.optim == "adagrad":
        optim = torch.optim.Adagrad(net.parameters(), opts.lr, weight_decay=weight_decay)
    elif opts.optim == "radam":
        optim = RAdam(net.parameters(), opts.lr, weight_decay=weight_decay)
    elif opts.optim == "ranger":
        optim = Ranger21(
            net.parameters(),
            opts.lr,
            weight_decay=weight_decay,
            lookahead_active=True,
            use_warmup=True,
            num_batches_per_epoch=len(train_loader),
            num_epochs=opts.n_epoch,
        )
    else:
        raise NotImplementedError

    if opts.lr_policy == "const":
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda x: 1)
    elif opts.lr_policy == "poly":
        lr_scheduler = PolynomialLRDecay(optim, opts.n_epoch, end_learning_rate=opts.lr * 0.1, power=0.9)
    elif opts.lr_policy == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, **opts.lr_policy_params)
    elif opts.lr_policy == "multistep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, **opts.lr_policy_params)
    elif opts.lr_policy == "plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim,
            mode="max",
            factor=0.1,
            patience=opts.lr_policy_params["patience"],
            cooldown=50,
            min_lr=1e-5,
        )
    elif opts.lr_policy == "SGDR":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optim,
            T_0=opts.lr_policy_params["T_0"],
            T_mult=opts.lr_policy_params["T_mult"],
            eta_min=opts.lr_policy_params["eta_min"],
        )
    else:
        raise NotImplementedError

    params = {
        "opts": opts,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "net": net,
        "optim": optim,
        "loss": loss,
        "lr_scheduler": lr_scheduler,
        "writer": writer,
        "valid_interval": valid_interval,
        "device": device,
        "model_dir": model_dir,
        "logger_name": f"{opts.tensor_dim}-Trainer",
        "multi_input_keys": multi_input_keys,
        "multi_output_keys": multi_output_keys,
    }

    engine = TRAIN_ENGINES[opts.framework](**params)

    return engine, net


def get_test_engine(opts, test_loader):
    """Generate engine for testing.

    Args:
        opts (SimpleNamespace): All arguments from cmd line.
        test_loader (DataLoader): Pytorch dataload for test dataset.

    Returns:
        IgniteEngine: Return test engine.
    """

    device = torch.device("cuda:0") if opts.gpus != "-1" else torch.device("cpu")

    frame, dim, data = opts.framework, opts.tensor_dim, opts.data_list
    multi_input_keys = DATASET_MAPPING[frame][dim][data].get("M_IN", None)
    multi_output_keys = DATASET_MAPPING[frame][dim][data].get("M_OUT", None)

    net = get_network(opts).to(device)

    params = {
        "opts": opts,
        "test_loader": test_loader,
        "net": net,
        "device": device,
        "logger_name": f"{opts.tensor_dim}-Tester",
        "multi_input_keys": multi_input_keys,
        "multi_output_keys": multi_output_keys,
        "output_latent_code": opts.save_latent,
        "target_latent_layer": opts.target_layer,
    }

    is_intra_ensemble = isinstance(opts.model_path, (list, tuple)) and len(opts.model_path) > 1

    if get_attr_(opts, "n_fold", 0) > 1 or get_attr_(opts, "n_repeat", 0) > 1 or is_intra_ensemble:
        return ENSEMBLE_TEST_ENGINES[frame](**params)
    else:
        return TEST_ENGINES[frame](**params)
