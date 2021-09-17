import time
import click
from pathlib import Path
from functools import partial, wraps
from click import option, prompt
from medlp.configures import config as cfg
from medlp.utilities.click_ex import (
    NumericChoice as Choice,
    lr_schedule_params,
    loss_select,
    model_select,
    data_select
)
from medlp.utilities.enum import *
from utils_cw import prompt_when


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
        return [files[selected]]


def common_params(func):
    @option("--tensor-dim", prompt=True, type=Choice(["2D", "3D"]),
            default='2D', help="2D or 3D")
    @option("--framework", prompt=True, type=Choice(FRAMEWORK_TYPES),
            default=1, help="Choose your framework type")
    @option("--data-list", type=str, callback=data_select,
            default=None, help="Data file list (json/yaml)")
    @option("--preload", type=float, default=1.0,
            help="Ratio of preload data")
    @option("--n-epoch", prompt=True, show_default=True,
            type=int, default=1000, help="Epoch number")
    @option("--n-epoch-len", type=float, default=1.0,
            help="Num of iterations for one epoch,"
                 "if n_epoch_len <= 1: n_epoch_len = n_epoch_len*n_epoch")
    @option("--n-batch", prompt=True, show_default=True,
            type=int, default=10, help="Batch size")
    @option("-IS", "--imbalance-sample", is_flag=True,
            help="Use imbalanced dataset sampling")
    @option("--downsample", type=int, default=-1,
            help="Downsample rate. disable:-1")
    @option("--smooth", type=float, default=0,
            help="Smooth rate, disable:0")
    @option("--input-nc", type=int, default=1,
            help="input data channels")
    @option("--output-nc", type=int, default=3,
            help="output channels (classes)")
    @option("--split", type=float, default=0.2,
            help="Training/testing split ratio")
    @option("--train-list", type=str, default='',
            help='Specified training datalist')
    @option("--valid-list", type=str, default='',
            help='Specified validation datalist')
    @option("-W", "--pretrained-model-path", type=str,
            default="", help="pretrained model path")
    @option("--out-dir", type=str, prompt=True, show_default=True,
            default=cfg.get_medlp_cfg('OUTPUT_DIR'))
    @option("--augment-ratio", type=float, default=0.3,
            help="Data augumentation ratio")
    @option("-P", "--partial", type=float,
            default=1, help="Only load part of data")
    @option("-V", "--visualize", is_flag=True,
            help="Visualize the network architecture")
    @option("--valid-interval", type=int, default=4,
            help="Interval of validation during training")
    @option("--early-stop", type=int, default=200,
            help="Patience of early stopping. default: 200epochs")
    @option("--save-epoch-freq", type=int, default=5,
            help="Save model freq")
    @option("--save-n-best", type=int, default=3,
            help="Save best N models")
    @option("--amp", is_flag=True,
            help="Flag of using amp. Need pytorch1.6")
    @option("--nni", is_flag=True,
            help="Flag of using nni-search, you dont need to modify this")
    @option("--n-fold", type=int, default=0,
            help="K fold cross-validation")
    @option("--n-repeat", type=int, default=0,
            help="K times random permutation cross-validator")
    @option("--ith-fold", type=int, default=-1,
            help="i-th fold of cross-validation")
    @option("--seed", type=int, default=101,
            help="random seed")
    @option("--compact-log", is_flag=True,
            help="Output compact log info")
    @option("--symbolic-tb", is_flag=True,
            help='Create symbolic for tensorboard logs')
    @option("--timestamp", type=str, default=time.strftime("%m%d_%H%M"),
            help="Timestamp")
    @option("--debug", is_flag=True,
            help='Enter debug mode')
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def solver_params(func):
    @option("--optim", type=Choice(OPTIM_TYPES), default="sgd",
            help='Optimizer for network')
    @option("--momentum", type=float, default=0.0,
            help="Momentum for optimizer")
    @option("--nesterov", type=bool, default=False,
            help="Nesterov for SGD")
    @option("-WD", "--l2-weight-decay", type=float, default=0, 
            help="weight decay (L2 penalty)")
    @option("--lr", type=float, default=1e-3,
            help="learning rate")
    @option("--lr-policy", prompt=True, type=Choice(LR_SCHEDULE),
            callback=lr_schedule_params, default="plateau", help="learning rate strategy")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def network_params(func):
    @option("--model-name", type=str, callback=model_select,
            default=None, help="Select deeplearning model")
    @option("-L", "--criterion", type=str, callback=loss_select,
            default=None, help="loss criterion type")
    @option("--layer-norm", prompt=True, type=Choice(NORM_TYPES),
            default=1, help="Layer norm type")
    @option("--layer-act", type=Choice(ACT_TYPES),
            default='relu', help="Layer activation type")
    @option("--n-features", type=int, default=64,
            help="Feature num of first layer")
    @option("--n-depth", type=int, default=-1,
            help="Network depth. -1: use default depth")
    @option("--feature-scale", type=int, default=4,
            help="not used")
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
    @option("--lr-policy-params", type=dict, default=None,
            help="Auxilary params for lr schedule")
    @option("--loss-params", type=dict, default={},
            help="Auxilary params for loss")
    @option("--pretrained", type=bool, default=False,
            help="Load pretrained model which have hard-coded model path")
    @option("--deep-supervision", type=bool, default=False,
            help="Use deep supervision module")
    @option("--deep-supr-num", type=int, default=1,
            help="Num of features will be output")
    @option("--snip-percent", type=float, default=0.4,
            callback=partial(prompt_when, keyword="snip"),
            help="Pruning ratio of wights/channels")
    @option("--config", type=click.Path(exists=True))
    @option("--n-group", type=int, default=1,
            help='Num of conv groups')
    # @option("--bott leneck-size", type=int, default=1, help='Size of bottleneck size of VGG net')
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def rcnn_params(func):
    @option("--model-type", type=Choice(RCNN_MODEL_TYPES),
            default=1, help="RCNN model type")
    @option("--backbone", type=Choice(RCNN_BACKBONE),
            default=1, help="RCNN backbone net")
    @option("--min-size", type=int, default=800)
    @option("--max-size", type=int, default=1000)
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
