import os, time, click
from types import SimpleNamespace as sn
from functools import partial, wraps
from click import Tuple
from medlp.utilities.click_ex import ChoiceEx as Choice
from medlp.utilities.click_ex import optionex, prompt_ex
from medlp.utilities.enum import *
from medlp.utilities.utils import is_avaible_size
from utils_cw import (
    Print,
    check_dir,
    prompt_when,
    recursive_glob2,
    get_items_from_file
    )


def get_trained_models(exp_folder):
    model_dir = os.path.join(exp_folder, "Models")
    assert os.path.isdir(model_dir), f"Model dir is not found! {model_dir}"
    files = recursive_glob2(model_dir, "*.pt", "*.pth", logic="or")
    prompt = {i: f.stem.split("=")[-1] for i, f in enumerate(files)}
    selected = prompt_ex(f"Choose model: {prompt}", type=int)
    return str(files[selected])


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

    # suffix = '-redo' if ctx.params.get('config') is not None else ''

    input_str = prompt_ex("Experiment name", default=exp_name, type=str)
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
    return prompt_ex(
        "\tInput {}".format(prompt_str),
        type=data_type,
        default=default_value,
        value_proc=value_proc,
        color=color
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


def loss_params(ctx, param, value):
    # if ctx.params.get('loss_params', (0,0)) is not (0,0): #loaded config from specified file
    #     return value

    if value == "WCE" or value == "WBCE":
        weights = _prompt("Loss weights", tuple, (0.1, 0.9), split_input_str_)
        ctx.params["loss_params"] = weights
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
        return prompt_ex("Model list", type=Choice(archilist, show_index=True))

    #! How to handle?
    # if value in ['vgg13', 'vgg16', 'resnet18', 'resnet34','resnet50']:
    #     ctx.params['load_imagenet'] = click.confirm("Whether load pretrained ImageNet model?", default=False, abort=False, show_default=True)
    #     if ctx.params['load_imagenet']:
    #         ctx.params['input_nc'] = 3
    # elif value == 'unet':
    #     ctx.params['deep_supervision'] = click.confirm("Whether use deep supervision?", default=False, abort=False, show_default=True)
    #     if ctx.params['deep_supervision']:
    #         ctx.params['deep_supr_num'] = prompt_ex("Num of deep supervision?", default=1, type=int, show_default=True)
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
        return prompt_ex("Data list", type=Choice(datalist, show_index=True))


def input_cropsize(ctx, param, value):
    configures = get_items_from_file(ctx.params["config"], format="json")
    if is_avaible_size(configures.get('crop_size', None)) or value is False:
        return value

    if configures['tensor_dim'] == '2D':
        crop_size = _prompt(
            "Crop size", tuple, (0, 0), partial(split_input_str_,dtype=int), color='green'
        )
    else:
        crop_size = _prompt(
            "Crop size", tuple, (0, 0, 0), partial(split_input_str_,dtype=int), color='green'
        )
    ctx.params["crop_size"] = crop_size
    return value


def common_params(func):
    @optionex(
        "--tensor-dim",
        prompt=True,
        type=Choice(["2D", "3D"], show_index=True),
        default=0,
        help="2D or 3D",
    )
    @optionex(
        "--framework",
        prompt=True,
        type=Choice(FRAMEWORK_TYPES, show_index=True),
        default=0,
        help="Choose your framework type",
    )
    @optionex(
        "--data-list",
        type=str,
        callback=data_select,
        default=None,
        help="Data file list (json)",
    )
    @optionex(
        "--preload", type=float, default=1.0, help="Ratio of preload data"
    )
    @optionex(
        "--n-epoch",
        prompt=True,
        show_default=True,
        type=int,
        default=1000,
        help="Epoch number",
    )
    @optionex(
        "--n-epoch-len",
        type=float,
        default=1.0,
        help="Num of iterations for one epoch, if n_epoch_len <= 1: n_epoch_len = n_epoch_len*n_epoch",
    )
    @optionex(
        "--n-batch",
        prompt=True,
        show_default=True,
        type=int,
        default=10,
        help="Batch size",
    )
    @optionex("--istrain", type=bool, default=True, help="train/test phase flag")
    @optionex("--downsample", type=int, default=-1, help="Downsample rate. disable:-1")
    @optionex("--smooth", type=float, default=0, help="Smooth rate, disable:0")
    @optionex("--input-nc", type=int, default=1, help="input data channels")
    @optionex("--output-nc", type=int, default=3, help="output channels (classes)")
    @optionex("--split", type=float, default=0.1, help="Training/testing split ratio")
    @optionex(
        "-W",
        "--pretrained-model-path",
        type=str,
        default="",
        help="pretrained model path",
    )
    @optionex(
        "--out-dir",
        type=str,
        prompt=True,
        show_default=True,
        default="/homes/clwang/Data/medlp_exp",
    )
    @optionex(
        "--augment-ratio", type=float, default=0.3, help="Data aug ratio."
    )
    @optionex(
        "-P",
        "--partial",
        type=float, default=1, help="Only load part of data"
    )
    @optionex(
        "-V",
        "--visualize",
        is_flag=True, help="Visualize the network architecture"
    )
    @optionex(
        "--valid-interval",
        type=int,
        default=4,
        help="Interval of validation during training",
    )
    @optionex(
        "--save-epoch-freq", type=int, default=5, help="Save model freq"
    )
    @optionex(
        "--early-stop",
        type=int,
        default=200,
        help="Patience of early stopping. default: 200epochs",
    )
    @optionex(
        "--amp", is_flag=True, help="Flag of using amp. Need pytorch1.6"
    )
    @optionex(
        "--nni",
        is_flag=True,
        help="Flag of using nni-search, you dont need to modify this.",
    )
    @optionex("--n-fold", type=int, default=0, help="K fold cross-validation")
    @optionex("--ith-fold", type=int, default=-1, help="i-th fold of cross-validation")
    @optionex("--seed", type=int, default=101, help="random seed")
    @optionex("--compact-log", is_flag=True, help="Output compact log info")
    @optionex("--symbolic-tb", is_flag=True, help='Create symbolic for tensorboard logs')
    @optionex(
        "--timestamp", type=str, default=time.strftime("%m%d_%H%M"), help="Timestamp"
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def solver_params(func):
    @optionex(
        "--optim", type=Choice(OPTIM_TYPES, show_index=True), default=1
    )
    @optionex(
        "--momentum", type=float, default=0.0, help="Momentum for optimizer"
    )
    @optionex(
        "-WD",
        "--l2-weight-decay",
        type=float,
        default=0,
        help="weight decay (L2 penalty)",
    )
    @optionex(
        "--lr", type=float, default=1e-3, help="learning rate"
    )
    @optionex(
        "--lr-policy",
        prompt=True,
        type=Choice(LR_SCHEDULE, show_index=True),
        callback=lr_schedule_params,
        default="plateau",
        help="learning rate strategy",
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def network_params(func):
    @optionex(
        "--model-name",
        type=str,
        callback=model_select,
        default=None,
        help="Select deeplearning model",
    )
    @optionex(
        "-L",
        "--criterion",
        prompt=True,
        type=Choice(LOSSES, show_index=True),
        callback=loss_params,
        default=0,
        help="loss criterion type",
    )
    @optionex(
        "--layer-norm",
        prompt=True,
        type=Choice(NORM_TYPES, show_index=True),
        default=0,
        help="Layer norm type",
    )
    @optionex(
        "--n-features", type=int, default=64, help="Feature num of first layer"
    )
    @optionex(
        "--n-depth", type=int, default=-1, help="Network depth. -1: use default depth"
    )
    @optionex(
        "--is-deconv", type=bool, default=False, help="use deconv or interplate"
    )
    @optionex(
        "--feature-scale", type=int, default=4, help="not used"
    )
    @optionex(
        "--snip", is_flag=True
    )
    # @optionex('--layer-order', prompt=True, type=Choice(LAYER_ORDERS,show_index=True), default=0, help='conv layer order')
    # @optionex('--bottleneck', type=bool, default=False, help='Use bottlenect achitecture')
    # @optionex('--sep-conv', type=bool, default=False, help='Use Depthwise Separable Convolution')
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# Put these auxilary params to the top of click.options for
# successfully loading auxilary params.
def latent_auxilary_params(func):
    @optionex(
        "--lr-policy-params",
        type=dict,
        default=None,
        help="Auxilary params for lr schedule",
    )
    @optionex(
        "--loss-params",
        type=(float, float),
        default=(0, 0),
        help="Auxilary params for loss",
    )
    @optionex(
        "--load-imagenet",
        type=bool,
        default=False,
        help="Load pretrain Imagenet for some net",
    )
    @optionex(
        "--deep-supervision",
        type=bool,
        default=False,
        help="Use deep supervision module",
    )
    @optionex(
        "--deep-supr-num",
        type=int,
        default=1,
        help="Num of features will be output"
    )
    @optionex(
        "--snip_percent",
        type=float,
        default=0.4,
        callback=partial(prompt_when, keyword="snip"),
        help="Pruning ratio of wights/channels",
    )
    @optionex("--n-fold", type=int, default=0)
    @optionex("--config", type=click.Path(exists=True))
    @optionex("--bottleneck-size", type=int, default=7, help='Size of bottleneck size of VGG net')
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def rcnn_params(func):
    @optionex(
        "--model-type",
        type=Choice(RCNN_MODEL_TYPES, show_index=True),
        default=0,
        help="RCNN model type",
    )
    @optionex(
        "--backbone",
        type=Choice(RCNN_BACKBONE, show_index=True),
        default=0,
        help="RCNN backbone net",
    )
    @optionex("--min-size", type=int, default=800)
    @optionex("--max-size", type=int, default=1000)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper