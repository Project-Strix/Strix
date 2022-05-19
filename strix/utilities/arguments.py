import time
from functools import partial, wraps
from pathlib import Path

import click
from click import prompt, UNPROCESSED
from strix.configures import config as cfg
from strix.utilities.click import OptionEx
from strix.utilities.click_callbacks import NumericChoice as Choice, framework_select
from strix.utilities.click_callbacks import (
    data_select, loss_select, lr_schedule_params, model_select, parse_input_str, multi_ouputnc, freeze_option
)
from strix.utilities.enum import ACTIVATIONS, FRAMEWORKS, LR_SCHEDULES, NORMS, OPTIMIZERS, FREEZERS
from utils_cw import prompt_when


option = partial(click.option, cls=OptionEx)

def get_best_trained_models(exp_folder, best_model_dirname: str = "Best_Models"):
    model_rootdir = Path(exp_folder)
    assert model_rootdir.is_dir(), f"Model dir is not found! {model_rootdir}"

    best_models = []

    for model_dir in list(model_rootdir.rglob(best_model_dirname)):
        files = list(filter(lambda x: x.suffix in [".pt", ".pth"], model_dir.iterdir()))
        if len(files) > 0:
            best_model = sorted(files, key=lambda x: float(x.stem.split("=")[-1]))[-1]
            best_models.append(best_model)
        else:
            continue
    return best_models


def get_trained_models(exp_folder):
    model_dir = Path(exp_folder) / "Models"
    assert model_dir.is_dir(), f"Model dir is not found! {model_dir}"

    subcategories = list(filter(lambda x: x.is_dir(), model_dir.iterdir()))

    off = 1
    prompt_1 = {i + off: f.stem for i, f in enumerate(subcategories)}
    selected = prompt(f"Choose model dir: {prompt_1}", type=int) - off

    files = list(filter(lambda x: x.suffix in [".pt", ".pth"], subcategories[selected].iterdir()))
    # files = recursive_glob2(model_dir, "*.pt", "*.pth", logic="or")
    prompt_2 = {i + off: f.stem.split("=")[-1] for i, f in enumerate(files)}
    if len(prompt_2) > 1:
        prompt_2.update({off - 1: "ensemble"})
    selected = prompt(f"Choose model: {prompt_2}", type=int) - off
    if selected == -1:
        return files
    return [files[selected]]


def common_params(func):
    @option("--tensor-dim", prompt=True, type=Choice(["2D", "3D"]), default="2D", help="2D or 3D")
    @option(
        "--framework", prompt=True, type=Choice(FRAMEWORKS), default=1,
        callback=framework_select, help="Choose your framework type"
    )
    @option("--data-list", type=str, callback=data_select, default=None, help="Data file list (json/yaml)")
    @option("--preload", type=float, default=1.0, help="Ratio of preload data")
    @option("--n-epoch", prompt=True, show_default=True, type=int, default=1000, help="Epoch number")
    @option(
        "--n-epoch-len", type=float, default=1.0,
        help="Num of iterations for one epoch, if n_epoch_len <= 1: n_epoch_len = n_epoch_len*n_epoch",
    )
    @option("--n-batch", prompt=True, show_default=True, type=int, default=10, help="Train batch size")
    @option("--n-batch-valid", prompt=True, show_default=True, type=int, default=5, help="Valid batch size")
    @option("--n-worker", type=int, default=10, help="Num of workers for training dataloader")
    @option("-IS", "--imbalance-sample", is_flag=True, help="Use imbalanced dataset sampling")
    @option("--downsample", type=int, default=-1, help="Downsample rate. disable:-1")
    @option("--input-nc", type=int, default=1, prompt=True, help="Input data channels")
    @option("--output-nc", type=UNPROCESSED, default=1, prompt=True, callback=multi_ouputnc, help="Output channels")
    @option("--split", type=float, default=0.2, help="Training/testing split ratio")
    @option("--train-list", type=click.Path(exists=True), default=None, help="Specified training datalist")
    @option("--valid-list", type=click.Path(exists=True), default=None, help="Specified validation datalist")
    @option("-W", "--pretrained-model-path", type=str, default="", help="pretrained model path")
    @option("--out-dir", type=str, prompt=True, show_default=True, default=cfg.get_strix_cfg("OUTPUT_DIR"))
    @option("--augment-ratio", type=float, default=0.3, help="Data augumentation ratio")
    @option("-P", "--partial", type=float, default=1, help="Only load part of data")
    @option("-V", "--visualize", is_flag=True, help="Visualize the network architecture")
    @option("--valid-interval", type=int, default=2, help="Interval of validation during training")
    @option("--early-stop", type=int, default=100, help="Patience of early stopping. default: 100epochs")
    @option("--save-epoch-freq", type=int, default=100, help="Save model freq")
    @option("--save-n-best", type=int, default=3, help="Save best N models")
    @option("--amp", is_flag=True, help="Flag of using amp. Need pytorch1.6")
    @option("--nni", is_flag=True, help="Flag of using nni-search, you dont need to modify this")
    @option("--n-fold", type=int, default=0, help="K fold cross-validation")
    @option("--n-repeat", type=int, default=0, help="K times random permutation cross-validator")
    @option("--ith-fold", type=int, default=-1, help="i-th fold of cross-validation")
    @option("--seed", type=int, default=101, help="random seed")
    @option("--disable-logfile", is_flag=True, help="Stop dump log file to local disk")
    @option("--compact-log", is_flag=True, help="Output compact log info")
    @option("--symbolic-tb", is_flag=True, help="Create symbolic for tensorboard logs")
    @option("--timestamp", type=str, default=time.strftime("%m%d_%H%M"), help="Timestamp")
    @option("--debug", is_flag=True, help="Enter debug mode")
    @option("--image-size", callback=partial(parse_input_str, dtype=int), help="Image size")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def solver_params(func):
    @option("--optim", type=Choice(OPTIMIZERS), default="sgd", help="Optimizer for network")
    @option("--momentum", type=float, default=0.0, help="Momentum for optimizer")
    @option("--nesterov", type=bool, default=False, help="Nesterov for SGD")
    @option("-WD", "--l2-weight-decay", type=float, default=0, help="weight decay (L2 penalty)")
    @option("--lr", type=float, default=1e-3, help="learning rate")
    @option(
        "--lr-policy", prompt=True, type=Choice(LR_SCHEDULES),
        callback=lr_schedule_params, default="plateau", help="learning rate strategy"
    )
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def network_params(func):
    @option("--model-name", type=str, callback=model_select, default=None, help="Select deeplearning model")
    @option("-L", "--criterion", type=UNPROCESSED, callback=loss_select, default=None, help="loss criterion type")
    @option("--layer-norm", prompt=True, type=Choice(NORMS), default=1, help="Layer norm type")
    @option("--layer-act", type=Choice(ACTIVATIONS), default="relu", help="Layer activation type")
    @option("--n-features", type=int, default=64, help="Feature num of first layer")
    @option("--n-depth", type=int, default=-1, help="Network depth. -1: use default depth")
    @option("--feature-scale", type=int, default=4, help="not used")
    @option("--snip", is_flag=True)
    @option("--freeze", is_flag=True, help="Freeze network layers")
    @option(
        "--freeze-kwargs", type=Choice(FREEZERS), default=None, prompt=True,
        prompt_cond=lambda ctx: ctx.params['freeze'], callback=freeze_option, help="Freeze kwargs"
    )
    # @optionex('--layer-order', prompt=True, type=Choice(LAYER_ORDERS), default=1, help='conv layer order')
    # @optionex('--bottleneck', type=bool, default=False, help='Use bottlenect achitecture')
    # @optionex('--sep-conv', type=bool, default=False, help='Use Depthwise Separable Convolution')
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# Put these auxilary params to the top of click.options for
# successfully loading auxilary params when `train-from-cfg`
def hidden_auxilary_params(func):
    @option("--lr-policy-params", type=dict, default=None, help="Auxilary params for lr schedule")
    @option("--loss-params", type=dict, default={}, help="Auxilary params for loss")
    @option("--loss-params-task1", type=dict, default={}, hidden=True, help="Auxilary params for task1's loss")
    @option("--loss-params-task2", type=dict, default={}, hidden=True, help="Auxilary params for task2's loss")
    @option("--pretrained", type=bool, default=False, help="Load pretrained model which have hard-coded model path")
    @option("--deep-supervision", type=bool, default=False, help="Use deep supervision module")
    @option("--deep-supr-num", type=int, default=1, help="Num of features will be output")
    @option(
        "--snip-percent", type=float, default=0.4, 
        callback=partial(prompt_when, keyword="snip"), help="Pruning ratio of wights",
    )
    @option("--config", type=click.Path(exists=True))
    @option("--n-group", type=int, default=1, help="Num of conv groups")
    @option("--do-test", type=bool, default=False, hidden=True, help="Automatically do test after training")
    @option("--subtask1", type=str, default=None, hidden=True, help="Subtask 1 in multitask framework")
    @option("--subtask2", type=str, default=None, hidden=True, help="Subtask 2 in multitask framework")
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper
