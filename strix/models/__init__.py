import torch
from strix import strix_datasets, strix_networks, strix_losses

from strix.models.cnn.utils import print_network, PolynomialLRDecay
from strix.models.cnn.layers.radam import RAdam
from strix.models.cnn.layers.ranger21 import Ranger21
from strix.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from strix.utilities.utils import get_attr_
from strix.utilities.imports import ModuleManager
from strix.utilities.enum import Frameworks
from strix.configures import config as cfg
from strix.models.cnn.cnn_nets import *
from strix.models.transformer.transformer_nets import *
from monai_ex.utils import WorkflowException, check_dir
from monai.optimizers.lr_scheduler import ExponentialLR, LinearLR


ModuleManager.import_all(cfg.get_strix_cfg("EXTERNAL_NETWORK_DIR"), recursive=True)
ModuleManager.import_all(cfg.get_strix_cfg("EXTERNAL_LOSS_DIR"), recursive=True)


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2**k for k in range(number_of_fmaps)]


def get_loss_fn(framework: str, loss_name: str, loss_params: dict, output_nc: int, deep_supervision: bool = False):
    if (loss_type := strix_losses.get(framework, loss_name)) is None:
        raise ValueError(f"Loss function {loss_name} is not found. It's weird!")

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

    model = strix_networks.get(opts.tensor_dim, opts.framework, opts.model_name)
    if model is None:
        raise ValueError(f"Cannot find registered model: {opts.model_name}")

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


def get_engine(opts, train_loader, test_loader, unlabel_loader=None, writer=None):
    """Generate engines for specified config.

    Args:
        opts (SimpleNamespace): All arguments from cmd line.
        train_loader (DataLoader): Pytorch dataload for training dataset.
        test_loader (DataLoader): Pytorch dataload for validation dataset.
        unlabel_loader (DataLoader): Pytorch dataload for unlabeled dataset.
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
    deep_supervision = get_attr_(opts, "deep_supervision", False)

    frame, dim, name = opts.framework, opts.tensor_dim, opts.data_list
    multi_input_keys = strix_datasets.get(dim, frame, name).get("M_IN", None)
    multi_output_keys = strix_datasets.get(dim, frame, name).get("M_OUT", None)

    device = torch.device("cuda") if opts.gpus != "-1" else torch.device("cpu")
    model_dir = check_dir(opts.experiment_path, "Models")

    loss = lr_scheduler = None
    if opts.framework == Frameworks.MULTITASK.value:
        subloss1 = get_loss_fn(
            opts.subtask1, opts.criterion[1], opts.loss_params_task1, opts.output_nc[0], deep_supervision
        )
        subloss2 = get_loss_fn(
            opts.subtask2, opts.criterion[2], opts.loss_params_task2, opts.output_nc[1], deep_supervision
        )
        loss = strix_losses.get(opts.framework, opts.criterion[0])(subloss1, subloss2, aggregate="sum", **opts.loss_params)
    else:
        loss = get_loss_fn(opts.framework, opts.criterion, opts.loss_params, opts.output_nc, deep_supervision)

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
    elif opts.lr_policy == "linear":
        lr_scheduler = LinearLR(optim, **opts.lr_policy_params)
    elif opts.lr_policy == "exponential":
        lr_scheduler = ExponentialLR(optim, **opts.lr_policy_params)
    else:
        raise NotImplementedError(f"Not supported LR scheduler {opts.lr_policy}")

    params = {
        "opts": opts,
        "train_loader": train_loader,
        "test_loader": test_loader,
        "unlabel_loader": unlabel_loader,
        "net": net,
        "optim": optim,
        "loss": loss,
        "lr_scheduler": lr_scheduler,
        "writer": writer,
        "valid_interval": valid_interval,
        "device": device,
        "model_dir": model_dir,
        "logger_name": None,
        "multi_input_keys": multi_input_keys,
        "multi_output_keys": multi_output_keys,
    }

    try:
        if unlabel_loader:
            engine = TRAIN_ENGINES["semi-" + opts.framework](**params)
        else:
            engine = TRAIN_ENGINES[opts.framework](**params)
    except Exception as e:
        raise WorkflowException() from e

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

    frame, dim, name = opts.framework, opts.tensor_dim, opts.data_list
    multi_input_keys = strix_datasets.get(dim, frame, name).get("M_IN", None)
    multi_output_keys = strix_datasets.get(dim, frame, name).get("M_OUT", None)

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
