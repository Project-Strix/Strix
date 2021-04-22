import os
import time
import click
from pathlib import Path
from types import SimpleNamespace as sn
from functools import partial, wraps
from termcolor import colored
from click import option, prompt
from medlp.configures import get_cfg
from medlp.utilities.click_ex import NumericChoice as Choice
from medlp.utilities.enum import *
from medlp.utilities.utils import is_avaible_size
from utils_cw import (
    Print,
    check_dir,
    prompt_when,
    get_items_from_file
    )


def get_trained_models(exp_folder, use_best_model=False):
    model_dir = Path(exp_folder)/"Models"
    assert model_dir.is_dir(), f"Model dir is not found! {model_dir}"

    subcategories = list(filter(lambda x: x.is_dir(), model_dir.iterdir()))
    if use_best_model:
        best_models = []
        for subdir in subcategories:
            files = list(filter(lambda x: x.suffix in [".pt", ".pth"], subdir.iterdir()))
            if len(files) > 0:
                best_model = sorted(files, key=lambda x: float(x.stem.split("=")[-1]))[-1]
                best_models.append(best_model)
            else:
                continue
        return best_models
    else:
        prompt_1 = {i: f.stem for i, f in enumerate(subcategories)}
        selected = prompt(f"Choose model dir: {prompt_1}", type=int)

        files = list(filter(lambda x: x.suffix in [".pt", ".pth"], subcategories[selected].iterdir()))
        # files = recursive_glob2(model_dir, "*.pt", "*.pth", logic="or")
        prompt_2 = {i: f.stem.split("=")[-1] for i, f in enumerate(files)}
        selected = prompt(f"Choose model: {prompt_2}", type=int)
        return [str(files[selected])]


def get_exp_name(ctx, param, value):
    model_name = ctx.params["model_name"]
    datalist_name = str(ctx.params["data_list"])
    partial_data = (
        "-partial" if "partial" in ctx.params and ctx.params["partial"] < 1 else ""
    )

    if ctx.params["lr_policy"] == "plateau" and ctx.params["valid_interval"] != 1:
        Print(
            "Warning: recommand set valid-interval = 1 when using ReduceLROnPlateau",
            color="y",
        )

    if "debug" in ctx.params and ctx.params["debug"]:
        Print("You are in Debug mode with preload=0, out_dir=test", color="y")
        ctx.params["preload"] = 0.0  # emmm...
        return check_dir(ctx.params["out_dir"], "test")

    mapping = {"batch": "BN", "instance": "IN", "group": "GN"}
    layer_norm = mapping[ctx.params["layer_norm"]]
    # update timestamp if train-from-cfg
    timestamp = (
        time.strftime("%m%d_%H%M")
        if ctx.params.get("config") is not None
        else ctx.params["timestamp"]
    )

    exp_name = (
        f"{model_name}-{ctx.params['criterion'].split('_')[0]}-"
        f"{layer_norm}-{ctx.params['optim']}-"
        f"{ctx.params['lr_policy']}{partial_data}-{timestamp}"
    )

    if ctx.params["n_fold"] > 0:
        exp_name = exp_name + f"-CV{ctx.params['n_fold']}"
    elif ctx.params["n_repeat"] > 0:
        exp_name = exp_name + f"-RE{ctx.params['n_repeat']}"

    input_str = prompt("Experiment name", default=exp_name, type=str)
    exp_name = exp_name + "-" + input_str.strip("+") if "+" in input_str else input_str

    return os.path.join(
        ctx.params["out_dir"], ctx.params["framework"], datalist_name, exp_name
    )


def get_nni_exp_name(ctx, param, value):
    param_list = get_items_from_file(ctx.params["param_list"], format="json")
    param_list["out_dir"] = ctx.params["out_dir"]
    param_list["timestamp"] = time.strftime("%m%d_%H%M")
    context_ = sn(**{"params": param_list})
    return get_exp_name(context_, param, value)


def split_input_str_(value, dtype=float):
    if value is not None:
        value = value.strip()
        if "," in value:
            sep = ","
        elif ";" in value:
            sep = ";"
        else:
            sep = " "
        return [dtype(s) for s in value.split(sep)]
    else:
        return None


def _prompt(prompt_str, data_type, default_value, value_proc=None, color=None):
    prompt_str = f"\tInput {prompt_str}"
    if color is not None:
        prompt_str = colored(prompt_str, color=color)

    return prompt(
        prompt_str,
        type=data_type,
        default=default_value,
        value_proc=value_proc,
    )


def lr_schedule_params(ctx, param, value):
    if (
        ctx.params.get("lr_policy_params", None) is not None
    ):  # loaded config from specified file
        return value

    if value == "step":
        iters = _prompt("step iter", int, 50)
        gamma = _prompt("step gamma", float, 0.1)
        ctx.params["lr_policy_params"] = {"step_size": iters, "gamma": gamma}
    elif value == 'multistep':
        steps = _prompt('steps', tuple, (100, 300), value_proc=lambda x: list(map(int, x.split(','))))
        gamma = _prompt("step gamma", float, 0.1)
        ctx.params["lr_policy_params"] = {"milestones": steps, "gamma": gamma}
        # print("lr_policy_params", ctx.params["lr_policy_params"])
    elif value == "SGDR":
        t0 = _prompt("SGDR T-0", int, 50)
        eta = _prompt("SGDR Min LR", float, 1e-4)
        tmul = _prompt("SGDR T-mult", int, 1)
        # dcay = _prompt('SGDR decay', float, 1)
        ctx.params["lr_policy_params"] = {"T_0": t0, "eta_min": eta, "T_mult": tmul}
    elif value == "plateau":
        patience = _prompt("patience", int, 80)
        ctx.params["lr_policy_params"] = {"patience": patience}
    elif value == "CLR":
        raise NotImplementedError

    return value


def loss_select(ctx, param, value):
    from medlp.models.cnn.losses import LOSS_MAPPING

    losslist = list(LOSS_MAPPING[ctx.params["framework"]].keys())

    assert len(losslist) > 0, f"No loss available for {ctx.params['framework']}! Abort!"
    if value is not None and value in losslist:
        return value
    else:
        value = prompt("Loss list", type=Choice(losslist))
        # if value in ['WCE', 'WBCE', 'WCE-DCE']:
        if 'WCE' in value:
            weights = _prompt("Loss weights", tuple, (0.9, 0.1), split_input_str_)
            ctx.params["loss_params"] = {"weight": weights}
        elif value == 'WBCE':
            pos_weight = _prompt("Pos weight", float, 2.0)
            ctx.params['loss_params'] = {'pos_weight': pos_weight}
        elif 'FocalLoss' in value:
            gamma = _prompt("Gamma", float, 2.0)
            ctx.params["loss_params"] = {'gamma': gamma}
        elif 'Contrastive' in value:
            margin = _prompt("Margin", float, 2.0)
            ctx.params["loss_params"] = {'margin': margin}
        else:
            ctx.params["loss_params"] = {}
        return value


def model_select(ctx, param, value):
    from medlp.models.cnn import ARCHI_MAPPING

    archilist = list(
        ARCHI_MAPPING[ctx.params["framework"]][ctx.params["tensor_dim"]].keys()
    )
    assert (
        len(archilist) > 0
    ), f"No architecture available for {ctx.params['tensor_dim']} {ctx.params['framework']} task! Abort!"

    if value is not None and value in archilist:
        return value
    else:
        return prompt("Model list", type=Choice(archilist))

    #! How to handle?
    # if value in ['vgg13', 'vgg16', 'resnet18', 'resnet34','resnet50']:
    #     ctx.params['load_imagenet'] = click.confirm("Whether load pretrained ImageNet model?", default=False, abort=False, show_default=True)
    #     if ctx.params['load_imagenet']:
    #         ctx.params['input_nc'] = 3
    # elif value == 'unet':
    #     ctx.params['deep_supervision'] = click.confirm("Whether use deep supervision?", default=False, abort=False, show_default=True)
    #     if ctx.params['deep_supervision']:
    #         ctx.params['deep_supr_num'] = prompt("Num of deep supervision?", default=1, type=int, show_default=True)
    # else:
    #     pass

    return value


def data_select(ctx, param, value):
    from medlp.data_io import DATASET_MAPPING

    datalist = list(
        DATASET_MAPPING[ctx.params["framework"]][ctx.params["tensor_dim"]].keys()
    )
    datalist = list(filter(lambda x: "fpath" not in x, datalist))  #! Not good
    assert (
        len(datalist) > 0
    ), f"No datalist available for {ctx.params['tensor_dim']} {ctx.params['framework']} task! Abort!"

    if value is not None and value in datalist:
        return value
    else:
        return prompt("Data list", type=Choice(datalist))


def input_cropsize(ctx, param, value):
    configures = get_items_from_file(ctx.params["config"], format="json")
    if is_avaible_size(configures.get('crop_size', None)) or value is False:
        return value

    if configures['tensor_dim'] == '2D':
        crop_size = _prompt(
            "Crop size", tuple, (0, 0),
            partial(split_input_str_, dtype=int), color='green'
        )
    else:
        crop_size = _prompt(
            "Crop size", tuple, (0, 0, 0),
            partial(split_input_str_, dtype=int), color='green'
        )
    ctx.params["crop_size"] = crop_size
    return value


def common_params(func):
    @option(
        "--tensor-dim", prompt=True, type=Choice(["2D", "3D"]),
        default='2D', help="2D or 3D",
    )
    @option(
        "--framework", prompt=True, type=Choice(FRAMEWORK_TYPES),
        default=1, help="Choose your framework type",
    )
    @option(
        "--data-list", type=str, callback=data_select,
        default=None, help="Data file list (json/yaml)",
    )
    @option(
        "--preload", type=float, default=1.0, help="Ratio of preload data"
    )
    @option(
        "--n-epoch", prompt=True, show_default=True, 
        type=int, default=1000, help="Epoch number",
    )
    @option(
        "--n-epoch-len", type=float, default=1.0,
        help="Num of iterations for one epoch, if n_epoch_len <= 1: n_epoch_len = n_epoch_len*n_epoch",
    )
    @option(
        "--n-batch", prompt=True, show_default=True,
        type=int, default=10, help="Batch size",
    )
    @option("--downsample", type=int, default=-1, help="Downsample rate. disable:-1")
    @option("--smooth", type=float, default=0, help="Smooth rate, disable:0")
    @option("--input-nc", type=int, default=1, help="input data channels")
    @option("--output-nc", type=int, default=3, help="output channels (classes)")
    @option("--split", type=float, default=0.1, help="Training/testing split ratio")
    @option("--train-list", type=str, default='', help='Specified training datalist')
    @option("--valid-list", type=str, default='', help='Specified validation datalist')
    @option(
        "-W", "--pretrained-model-path",
        type=str, default="", help="pretrained model path",
    )
    @option(
        "--out-dir", type=str, prompt=True,
        show_default=True, default=get_cfg('MEDLP_CONFIG', 'OUTPUT_DIR'),
    )
    @option("--augment-ratio", type=float, default=0.3, help="Data aug ratio")
    @option(
        "-P", "--partial",
        type=float, default=1, help="Only load part of data"
    )
    @option(
        "-V", "--visualize",
        is_flag=True, help="Visualize the network architecture"
    )
    @option(
        "--valid-interval",
        type=int, default=4,
        help="Interval of validation during training",
    )
    @option(
        "--early-stop", type=int, default=200,
        help="Patience of early stopping. default: 200epochs",
    )
    @option("--save-epoch-freq", type=int, default=5, help="Save model freq")
    @option("--save-n-best", type=int, default=3, help="Save best N models")
    @option("--amp", is_flag=True, help="Flag of using amp. Need pytorch1.6")
    @option("--nni", is_flag=True, help="Flag of using nni-search, you dont need to modify this")
    @option("--n-fold", type=int, default=0, help="K fold cross-validation")
    @option("--n-repeat", type=int, default=0, help="K times random permutation cross-validator")
    @option("--ith-fold", type=int, default=-1, help="i-th fold of cross-validation")
    @option("--seed", type=int, default=101, help="random seed")
    @option("--compact-log", is_flag=True, help="Output compact log info")
    @option("--symbolic-tb", is_flag=True, help='Create symbolic for tensorboard logs')
    @option("--timestamp", type=str, default=time.strftime("%m%d_%H%M"), help="Timestamp")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def solver_params(func):
    @option("--optim", type=Choice(OPTIM_TYPES), default="sgd")
    @option("--momentum", type=float, default=0.0, help="Momentum for optimizer")
    @option("-WD", "--l2-weight-decay", type=float, default=0, help="weight decay (L2 penalty)")
    @option("--lr", type=float, default=1e-3, help="learning rate")
    @option("--lr-policy", prompt=True, type=Choice(LR_SCHEDULE), callback=lr_schedule_params, default="plateau", help="learning rate strategy")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def network_params(func):
    @option(
        "--model-name",
        type=str, callback=model_select,
        default=None, help="Select deeplearning model",
    )
    @option(
        "-L", "--criterion",
        type=str, callback=loss_select,
        default=None, help="loss criterion type",
    )
    @option(
        "--layer-norm",
        prompt=True, type=Choice(NORM_TYPES),
        default=1, help="Layer norm type",
    )
    @option("--n-features", type=int, default=64, help="Feature num of first layer")
    @option("--n-depth", type=int, default=-1, help="Network depth. -1: use default depth")
    @option("--is-deconv", type=bool, default=False, help="use deconv or interplate")
    @option("--feature-scale", type=int, default=4, help="not used")
    @option("--snip", is_flag=True)
    # @optionex('--layer-order', prompt=True, type=Choice(LAYER_ORDERS), default=1, help='conv layer order')
    # @optionex('--bottleneck', type=bool, default=False, help='Use bottlenect achitecture')
    # @optionex('--sep-conv', type=bool, default=False, help='Use Depthwise Separable Convolution')
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# Put these auxilary params to the top of click.options for
# successfully loading auxilary params.
def latent_auxilary_params(func):
    @option(
        "--lr-policy-params", type=dict,
        default=None, help="Auxilary params for lr schedule",
    )
    @option(
        "--loss-params", type=(float, float),
        default=(0, 0), help="Auxilary params for loss",
    )
    @option(
        "--load-imagenet", type=bool,
        default=False, help="Load pretrain Imagenet for some net",
    )
    @option(
        "--deep-supervision", type=bool,
        default=False, help="Use deep supervision module",
    )
    @option(
        "--deep-supr-num", type=int,
        default=1, help="Num of features will be output"
    )
    @option(
        "--snip-percent", type=float, default=0.4,
        callback=partial(prompt_when, keyword="snip"), help="Pruning ratio of wights/channels",
    )
    @option("--n-fold", type=int, default=0)
    @option("--config", type=click.Path(exists=True))
    @option("--bottleneck-size", type=int, default=1, help='Size of bottleneck size of VGG net')
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def rcnn_params(func):
    @option(
        "--model-type", type=Choice(RCNN_MODEL_TYPES),
        default=1, help="RCNN model type",
    )
    @option(
        "--backbone", type=Choice(RCNN_BACKBONE),
        default=1, help="RCNN backbone net",
    )
    @option("--min-size", type=int, default=800)
    @option("--max-size", type=int, default=1000)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
