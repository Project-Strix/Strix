from typing import Callable
import re
import os
import subprocess
from time import strftime
from pathlib import Path
import inspect
from functools import partial
from click import Choice, ParamType, prompt
import click
from click.types import convert_type
from types import SimpleNamespace as sn
from termcolor import colored
from medlp.data_io import DATASET_MAPPING
from medlp.utilities.utils import is_avaible_size
from utils_cw import Print, check_dir, get_items_from_file


###################### Extension of click ################################


class DynamicTuple(ParamType):
    def __init__(self, input_type):
        self.type = convert_type(input_type)

    @property
    def name(self):
        return "< Dynamic Tuple >"

    def convert(self, value, param, ctx):
        # Hotfix for prompt input
        if isinstance(value, str):
            if "," in value:
                sep = ","
            elif ";" in value:
                sep = ";"
            else:
                sep = " "

            value = value.strip().split(sep)
            value = list(filter(lambda x: x != " ", value))
        elif value is None or value == "":
            return None

        types = (self.type,) * len(value)
        return tuple(ty(x, param, ctx) for ty, x in zip(types, value))


class NumericChoice(Choice):
    def __init__(self, choices, **kwargs):
        self.choicemap = {}
        choicestrs = []
        for i, choice in enumerate(choices, start=1):
            self.choicemap[i] = choice
            if len(choices) > 5:
                choicestrs.append(f"\n\t{i}: {choice}")
            else:
                choicestrs.append(f"{i}: {choice}")

        super().__init__(choicestrs, **kwargs)

    def convert(self, value, param, ctx):
        try:
            return self.choicemap[int(value)]
        except ValueError as e:
            if value in self.choicemap.values():
                return value
            self.fail(
                f"invaid index choice: {value}. Please input integer index or correct value!"
                f"Error msg: {e}"
            )
        except KeyError as e:
            self.fail(
                f"invalid choice: {value}. (choose from {self.choicemap})", param, ctx
            )


#######################################################################

def select_gpu(ctx, parma, value):
    if value is not None:
        return value

    # check mig status
    MIG_CMD = ["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader"]
    LIST_CMD = ["nvidia-smi", "-L"]
    statuses, gpu_list = [], ''

    try:
        result = subprocess.check_output(MIG_CMD)
    except subprocess.CalledProcessError:
        pass
    else:
        modes = result.decode('utf-8').split('\n')
        statuses = ["enabled" == status.lower() for status in modes if status]
        gpu_list = subprocess.check_output(LIST_CMD).decode('utf-8').split('\n')
    finally:
        gpu_name = lambda x: f"GPU {x}"
        if any(statuses):
            choice_type: Callable = NumericChoice
            mig_mem_uuid = re.compile(r"(\d+g)b.*(MIG-[^)]*)")
            gpu_id = re.compile(r"GPU (\d)")
            mig_map = {gpu_name(i): mode for i, mode in enumerate(statuses)}
            gpu_tables = {}
            index = 0
            for gpu_str in gpu_list:
                gpu_str_ = gpu_str.strip()
                gid = gpu_id.search(gpu_str_)
                if gid:
                    index = int(gid.group(1))
                    if not mig_map[gpu_name(index)]:
                        gpu_tables.update({gpu_name(index): index})
                else:
                    mem_uuid = mig_mem_uuid.search(gpu_str_)
                    if mem_uuid:
                        gid = mem_uuid.group(2).split('/')[-2]
                        gpu_tables.update({gpu_name(index)+f"-{gid} ({mem_uuid.group(1)})": mem_uuid.group(2)})
        else:
            gpu_tables = {str(i): str(i) for i, s in enumerate(statuses)}
            choice_type: Callable = Choice

        selected_idx = prompt("Choose GPU from", type=choice_type(gpu_tables.keys()))
        return gpu_tables[selected_idx]


def _convert_type(var, types=[float, str]):
    for type_ in types:
        try:
            return type_(var)
        except ValueError as e:
            pass
    return var


def get_unknown_options(ctx, verbose=False):
    auxilary_params = {}

    if isinstance(ctx, sn):  #! temp solution
        return auxilary_params

    for i in range(0, len(ctx.args), 2):  # Todo: how to handle flag auxilary params?
        if str(ctx.args[i]).startswith("--"):
            auxilary_params[ctx.args[i][2:].replace("-", "_")] = _convert_type(
                ctx.args[i + 1]
            )
        elif str(ctx.args[i]).startswith("-"):
            auxilary_params[ctx.args[i][1:].replace("-", "_")] = _convert_type(
                ctx.args[i + 1]
            )
        else:
            Print("Got invalid argument:", ctx.args[i], color="y", verbose=verbose)

    Print("Got auxilary params:", auxilary_params, color="y", verbose=verbose)
    return auxilary_params


def get_exp_name(ctx, param, value):
    model_name = ctx.params["model_name"]
    datalist_name = str(ctx.params["data_list"])
    partial_data = (
        "-partial" if "partial" in ctx.params and ctx.params["partial"] < 1 else ""
    )

    if ctx.params["lr_policy"] == "plateau" and ctx.params["valid_interval"] != 1:
        Print(
            "Warning: recommand set valid-interval = 1" "when using ReduceLROnPlateau",
            color="y",
        )

    if "debug" in ctx.params and ctx.params["debug"]:
        Print("You are in Debug mode with preload=0, out_dir=debug", color="y")
        ctx.params["preload"] = 0.0  # emmm...
        return check_dir(ctx.params["out_dir"], "debug")

    mapping = {"batch": "BN", "instance": "IN", "group": "GN"}
    layer_norm = mapping[ctx.params["layer_norm"]]
    # update timestamp if train-from-cfg
    timestamp = (
        strftime("%m%d_%H%M")
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

    auxilary_params = get_unknown_options(ctx)
    if len(auxilary_params) > 0:
        exp_name = exp_name + "-" + "-".join(auxilary_params)

    input_str = prompt("Experiment name", default=exp_name, type=str)
    exp_name = exp_name + "-" + input_str.strip("+") if "+" in input_str else input_str

    project_name = DATASET_MAPPING[ctx.params["framework"]][ctx.params["tensor_dim"]][
        datalist_name
    ].get("PROJECT")

    if project_name:
        proj_dirname = f"Project-{project_name}"
        return (
            Path(ctx.params["out_dir"])
            / ctx.params["framework"]
            / proj_dirname
            / datalist_name
            / exp_name
        )
    else:
        return (
            Path(ctx.params["out_dir"])
            / ctx.params["framework"]
            / datalist_name
            / exp_name
        )


def get_nni_exp_name(ctx, param, value):
    param_list = get_items_from_file(ctx.params["param_list"], format="json")
    param_list["out_dir"] = ctx.params["out_dir"]
    param_list["timestamp"] = strftime("%m%d_%H%M")
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


def parse_input_str(ctx, param, value, dtype=str):
    return split_input_str_(value, dtype=dtype)


def _prompt(prompt_str, data_type, default_value, value_proc=None, color=None):
    prompt_str = f"\tInput {prompt_str} ({data_type})"
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
    elif value == "multistep":
        steps = _prompt(
            "steps",
            tuple,
            (100, 300),
            value_proc=lambda x: list(map(int, x.split(","))),
        )
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
        patience = _prompt("patience", int, 50)
        ctx.params["lr_policy_params"] = {"patience": patience}
    elif value == "CLR":
        raise NotImplementedError

    return value


def loss_select(ctx, param, value, prompt_all_args=False):
    from medlp.models.cnn.losses import LOSS_MAPPING

    losslist = list(LOSS_MAPPING[ctx.params["framework"]].keys())

    assert len(losslist) > 0, f"No loss available for {ctx.params['framework']}! Abort!"
    if value is not None and value in losslist:
        return value
    else:
        value = prompt("Loss list", type=NumericChoice(losslist))

        if "WCE" in value:
            weights = _prompt("Loss weights", tuple, (0.9, 0.1), split_input_str_)
            ctx.params["loss_params"] = {"weight": weights}
        elif value == "WBCE":
            pos_weight = _prompt("Pos weight", float, 2.0)
            ctx.params["loss_params"] = {"pos_weight": pos_weight}
        elif "FocalLoss" in value:
            gamma = _prompt("Gamma", float, 2.0)
            ctx.params["loss_params"] = {"gamma": gamma}
        elif "Contrastive" in value:
            margin = _prompt("Margin", float, 2.0)
            ctx.params["loss_params"] = {"margin": margin}
        elif value == "GDL":
            w_type = _prompt("Weight type(square, simple, uniform)", str, "square")
            ctx.params["loss_params"] = {"w_type": w_type}
        else:  #! custom loss
            func = LOSS_MAPPING[ctx.params["framework"]][value]
            sig = inspect.signature(func)
            anno = inspect.getfullargspec(func).annotations
            if not prompt_all_args:
                cond = lambda x: x[1].default is x[1].empty
            else:
                cond = lambda x: True

            loss_params = {}
            for k, v in filter(cond, sig.parameters.items()):
                if anno.get(k) in BUILTIN_TYPES:
                    default_value = None if v.default is v.empty else v.default
                    loss_params[k] = _prompt(
                        f"Loss argument '{k}'", anno[k], default_value
                    )
                else:
                    print(f"Cannot handle type '{anno.get(k)}' for argment '{k}'")
                    # raise ValueError(f"Cannot handle type '{anno.get(k)}' for argment '{k}'")
            ctx.params["loss_params"] = loss_params

        return value


def model_select(ctx, param, value):
    from medlp.models import ARCHI_MAPPING

    archilist = list(
        ARCHI_MAPPING[ctx.params["framework"]][ctx.params["tensor_dim"]].keys()
    )
    assert (
        len(archilist) > 0
    ), f"No architecture available for {ctx.params['tensor_dim']} {ctx.params['framework']} task! Abort!"

    if value is not None and value in archilist:
        return value
    else:
        return prompt("Model list", type=NumericChoice(archilist))

    return value


def data_select(ctx, param, value):
    from medlp.data_io import DATASET_MAPPING

    datalist = list(
        DATASET_MAPPING[ctx.params["framework"]][ctx.params["tensor_dim"]].keys()
    )

    assert len(datalist) > 0, (
        f"No datalist available for {ctx.params['tensor_dim']} "
        f"{ctx.params['framework']} task! Abort!"
    )

    if value is not None and value in datalist:
        return value
    else:
        return prompt("Data list", type=NumericChoice(datalist))


def input_cropsize(ctx, param, value):
    if value is False:
        return value

    configures = get_items_from_file(ctx.params["config"], format="json")
    if is_avaible_size(configures.get("crop_size", None)) or value is False:
        return value

    if configures["tensor_dim"] == "2D":
        crop_size = _prompt(
            "Crop size",
            tuple,
            (0, 0),
            partial(split_input_str_, dtype=int),
            color="green",
        )
    else:
        crop_size = _prompt(
            "Crop size",
            tuple,
            (0, 0, 0),
            partial(split_input_str_, dtype=int),
            color="green",
        )
    ctx.params["crop_size"] = crop_size
    return value


def check_batchsize(ctx_params):
    if "valid_list" in ctx_params and os.path.isfile(ctx_params["valid_list"]):
        files_valid = get_items_from_file(ctx_params["valid_list"], format="auto")
        len_valid = len(files_valid)
    else:
        data_list = DATASET_MAPPING[ctx_params["framework"]][ctx_params["tensor_dim"]][ctx_params["data_list"]].get(
            "PATH", ""
        )
        split = ctx_params["split"]
        all_data_list = get_items_from_file(data_list, format="auto")
        len_valid = len(all_data_list) * split

    if len_valid < ctx_params['n_batch_valid']:
        print(f"Validation Batch size {ctx_params['n_batch_valid']} larger than all valid data size {len_valid}!")
        raise click.Abort()

