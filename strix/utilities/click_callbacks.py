from email.policy import default
from typing import Callable

import re
import math
import json
import inspect
import subprocess
from functools import partial
from pathlib import Path
from time import strftime
from types import SimpleNamespace as sn
from termcolor import colored

from click import Choice, prompt
from strix.data_io import DATASET_MAPPING
from strix.models import ARCHI_MAPPING
from strix.models.cnn.losses import LOSS_MAPPING
from strix.utilities.enum import BUILTIN_TYPES, FRAMEWORKS, Frameworks
from strix.utilities.utils import is_avaible_size, get_items
from strix.utilities.enum import BUILTIN_TYPES
from strix.utilities.click import NumericChoice
from utils_cw import Print, check_dir, PathlibEncoder


#######################################################################

def _get_prompt_flag(ctx, param, value, sub_option_keyword=None):
    keyword = sub_option_keyword if sub_option_keyword else param.name
    remeber_mode = ctx.default_map is not None and ctx.prompt_in_default_map
    if sub_option_keyword and remeber_mode:  #like subtask option
        return ctx.meta.get(keyword) is None # avoid callback fn fires twice github.com/pallets/click/issues/2259
    elif ctx.default_map is None:
        default_value = ctx.params.get(keyword)
    else:
        default_value = ctx.default_map.get(keyword, ctx.params.get(keyword))

    prompt_flag = remeber_mode or (not remeber_mode and default_value is None)
    return prompt_flag

def select_gpu(ctx, param, value):
    if not _get_prompt_flag(ctx, param, value):
        return value

    # check mig status
    MIG_CMD = ["nvidia-smi", "--query-gpu=mig.mode.current", "--format=csv,noheader"]
    LIST_CMD = ["nvidia-smi", "-L"]
    statuses, gpu_list = [], ""

    try:
        result = subprocess.check_output(MIG_CMD)
    except subprocess.CalledProcessError:
        pass
    else:
        modes = result.decode("utf-8").split("\n")
        statuses = ["enabled" == status.lower() for status in modes if status]
        gpu_list = subprocess.check_output(LIST_CMD).decode("utf-8").split("\n")
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
                        gid = mem_uuid.group(2).split("/")[-2]
                        gpu_tables.update({gpu_name(index) + f"-{gid} ({mem_uuid.group(1)})": mem_uuid.group(2)})
        else:
            gpu_tables = {str(i): str(i) for i, s in enumerate(statuses)}
            choice_type: Callable = Choice

        selected_idx = prompt("Choose GPU from", type=choice_type(gpu_tables.keys()), default=None)
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
            auxilary_params[ctx.args[i][2:].replace("-", "_")] = _convert_type(ctx.args[i + 1])
        elif str(ctx.args[i]).startswith("-"):
            auxilary_params[ctx.args[i][1:].replace("-", "_")] = _convert_type(ctx.args[i + 1])
        else:
            Print("Got invalid argument:", ctx.args[i], color="y", verbose=verbose)

    Print("Got auxilary params:", auxilary_params, color="y", verbose=verbose)
    return auxilary_params


def get_exp_name(ctx, param, value):
    model_name = ctx.params["model_name"]
    datalist_name = str(ctx.params["data_list"])
    partial_data = "-partial" if "partial" in ctx.params and ctx.params["partial"] < 1 else ""

    if "debug" in ctx.params and ctx.params["debug"]:
        Print("You are in Debug mode with preload=0, out_dir=debug, n_worker=0", color="y")
        ctx.params["preload"] = 0.0  # emmm...
        ctx.params["n_worker"] = 0
        return check_dir(ctx.params["out_dir"], "debug")

    mapping = {"batch": "BN", "instance": "IN", "group": "GN"}
    layer_norm = mapping[ctx.params["layer_norm"]]
    # update timestamp if train-from-cfg
    timestamp = strftime("%m%d_%H%M") if ctx.params.get("config") is not None else ctx.params["timestamp"]

    if ctx.params["framework"] == Frameworks.MULTITASK.value:
        loss_fn = '_'.join(ctx.params['criterion'])
    else:
        loss_fn = ctx.params['criterion'].split('_')[0]

    exp_name = (
        f"{model_name}-{loss_fn}-{layer_norm}-{ctx.params['optim']}-"
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

    project_name = DATASET_MAPPING[ctx.params["framework"]][ctx.params["tensor_dim"]][datalist_name].get("PROJECT")

    if project_name and not project_name.startswith("Project"):
        proj_dirname = f"Project-{project_name}"
        return Path(ctx.params["out_dir"]) / ctx.params["framework"] / proj_dirname / datalist_name / exp_name
    else:
        return Path(ctx.params["out_dir"]) / ctx.params["framework"] / datalist_name / exp_name


def get_nni_exp_name(ctx, param, value):
    param_list = get_items(ctx.params["param_list"], format="json")
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
    if not _get_prompt_flag(ctx, param, value, "lr_policy_params"):  # loaded config from specified file
        return value

    if value == "step":
        iters = _prompt("step iter", int, 50)
        gamma = _prompt("step gamma", float, 0.1)
        ctx.params["lr_policy_params"] = ctx.meta["lr_policy_params"] = {"step_size": iters, "gamma": gamma}
    elif value == "multistep":
        steps = _prompt("steps", tuple, (100, 300), value_proc=lambda x: list(map(int, x.split(","))))
        gamma = _prompt("step gamma", float, 0.1)
        ctx.params["lr_policy_params"] = ctx.meta["lr_policy_params"] = {"milestones": steps, "gamma": gamma}
    elif value == "SGDR":
        t0 = _prompt("SGDR T-0", int, 50)
        eta = _prompt("SGDR Min LR", float, 1e-4)
        tmul = _prompt("SGDR T-mult", int, 1)
        # dcay = _prompt('SGDR decay', float, 1)
        ctx.params["lr_policy_params"] = ctx.meta["lr_policy_params"] = {"T_0": t0, "eta_min": eta, "T_mult": tmul}
    elif value == "plateau":
        patience = _prompt("patience", int, 50)
        ctx.params["lr_policy_params"] = ctx.meta["lr_policy_params"] = {"patience": patience}
    elif value == "CLR":
        raise NotImplementedError

    return value


def multi_ouputnc(ctx, param, value):
    if isinstance(value, str) and value.isnumeric():
        value = int(value)

    if ctx.params['framework'] != Frameworks.MULTITASK.value:
        return value

    if ctx.params['framework'] == Frameworks.MULTITASK.value and\
       not isinstance(value, list):
        subtask1_nc = prompt(f"Output nc for {colored('task1', 'yellow')}", type=int, default=value)
        subtask2_nc = prompt(f"Output nc for {colored('task2', 'yellow')}", type=int, default=value)
        return [subtask1_nc, subtask2_nc]
    else:
        return value


def framework_select(ctx, param, value):
    prompt_subtask = _get_prompt_flag(ctx, param, value, "subtask1")

    if value == Frameworks.MULTITASK.value and prompt_subtask:
        if "multitask" in FRAMEWORKS:
            FRAMEWORKS.remove("multitask")
        
        if ctx.default_map:
            default_subtask1 = ctx.default_map.get("subtask1", ctx.params.get('subtask1', 2))
            default_subtask2 = ctx.default_map.get("subtask2", ctx.params.get('subtask2', 1))
        else:
            default_subtask1 = ctx.params.get('subtask1', 2)
            default_subtask2 = ctx.params.get('subtask2', 1)

        subtask1 = prompt(
            f"Sub {colored('task1', 'yellow')}", type=NumericChoice(FRAMEWORKS), default=default_subtask1
        )
        subtask2 = prompt(
            f"Sub {colored('task2', 'yellow')}", type=NumericChoice(FRAMEWORKS), default=default_subtask2
        )
        ctx.params['subtask1'] = ctx.meta['subtask1'] = subtask1
        ctx.params['subtask2'] = ctx.meta['subtask2'] = subtask2

    return value


def loss_select(ctx, param, value, prompt_all_args=False):

    def _single_loss_select(ctx, framework, value, prompt_all_args, meta_key_postfix=''):
        losslist = list(LOSS_MAPPING[framework].keys())

        assert len(losslist) > 0, f"No loss available for {framework}! Abort!"

        if not _get_prompt_flag(ctx, param, value):
        # if value is not None and value in losslist:
            return value

        prompts = f"Loss_fn for {colored(meta_key_postfix[1:], 'yellow')}" if meta_key_postfix else "Loss_fn"
        value = prompt(prompts, type=NumericChoice(losslist), default=1)

        meta_key = f"loss_params{meta_key_postfix}"

        if "WCE" in value:
            weights = _prompt("Loss weights", tuple, (0.9, 0.1), split_input_str_)
            ctx.params[meta_key] = {"weight": weights}
        elif value == "WBCE":
            pos_weight = _prompt("Pos weight", float, 2.0)
            ctx.params[meta_key] = {"pos_weight": pos_weight}
        elif "FocalLoss" in value:
            gamma = _prompt("Gamma", float, 2.0)
            ctx.params[meta_key] = {"gamma": gamma}
        elif "Contrastive" in value:
            margin = _prompt("Margin", float, 2.0)
            ctx.params[meta_key] = {"margin": margin}
        elif value == "GDL":
            w_type = _prompt("Weight type(square, simple, uniform)", str, "square")
            ctx.params[meta_key] = {"w_type": w_type}
        elif value == "Weighted":
            weight = _prompt("Weight (task1/task2)", float, 1.0)
            ctx.params[meta_key] = {"weight": weight}
        elif value == "Uniform":
            ctx.params[meta_key] = {"weight": 1}
        else:  #! custom loss
            func = LOSS_MAPPING[framework][value]
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
                    loss_params[k] = _prompt(f"Loss argument '{k}'", anno[k], default_value)
                else:
                    print(f"Cannot handle type '{anno.get(k)}' for argment '{k}'")
                    # raise ValueError(f"Cannot handle type '{anno.get(k)}' for argment '{k}'")
            ctx.params[meta_key] = loss_params

        return value

    #Todo: need refactor this part
    if ctx.params["framework"] == Frameworks.MULTITASK.value:
        force_prompt = ctx.default_map is not None and ctx.prompt_in_default_map
        if not isinstance(value, list) or force_prompt:
            value0 = value[0] if isinstance(value, list) else value
            loss = _single_loss_select(ctx, ctx.params["framework"], value0, prompt_all_args)

            value1 = value[1] if isinstance(value, list) else value
            loss1 = _single_loss_select(ctx, ctx.params["subtask1"], value1, prompt_all_args, '_task1')
            value2 = value[2] if isinstance(value, list) else value
            loss2 = _single_loss_select(ctx, ctx.params["subtask2"], value2, prompt_all_args, '_task2')
            return [loss, loss1, loss2]
        else:
            return value
    else:
        return _single_loss_select(ctx, ctx.params["framework"], value, prompt_all_args)


def model_select(ctx, param, value):
    archilist = list(
        ARCHI_MAPPING[ctx.params["framework"]][ctx.params["tensor_dim"]].keys()
    )
    if len(archilist) == 0:
        print(
            f"No architecture available for {ctx.params['tensor_dim']} "
            f"{ctx.params['framework']} task! Abort!"
        )
        ctx.exit()

    if value is not None and value in archilist:
        return value
    else:
        return prompt("Model list", type=NumericChoice(archilist))


def data_select(ctx, param, value):
    datalist = list(
        DATASET_MAPPING[ctx.params["framework"]][ctx.params["tensor_dim"]].keys()
    )

    if len(datalist) == 0:
        print(
            f"No datalist available for {ctx.params['tensor_dim']} "
            f"{ctx.params['framework']} task! Abort!"
        )
        ctx.exit()

    if _get_prompt_flag(ctx, param, value):
        return prompt("Data list", type=NumericChoice(datalist))  #default="1"
    else:
        return value


def input_cropsize(ctx, param, value):
    if value is False:
        return value

    configures = get_items(ctx.params["config"], format="json")
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


def dump_params(ctx, param, value, output_path=None, skip_flag=True):
    if output_path:
        dump_dict = ctx.params.copy()
        with open(output_path, 'w') as f:
            for flag in [p.name for p in ctx.command.params if p.is_flag and p.name in dump_dict]:
                    dump_dict.pop(flag)
            json.dump(dump_dict, f, indent=2, sort_keys=True, cls=PathlibEncoder)
    return value

#######################################################################
##                    checklist callbacks
#######################################################################


def check_batchsize(ctx_params):
    train_list, valid_list, framework, tensor_dim, data_name, split, n_batch_train, n_batch_valid = (
        ctx_params.get("train_list"),
        ctx_params.get("valid_list"),
        ctx_params.get("framework"),
        ctx_params.get("tensor_dim"),
        ctx_params.get("data_list"),
        ctx_params.get("split"),
        ctx_params.get("n_batch"),
        ctx_params.get("n_batch_valid"),
    )
    if train_list and valid_list:
        print(valid_list, train_list)
        files_train = get_items(train_list, format="auto")
        files_valid = get_items(valid_list, format="auto")
        len_train, len_valid = len(files_train), len(files_valid)
    elif data_name in ["RandomData", "SyntheticData"]:
        all_cases = 100
        len_valid = int(all_cases * split)
        len_train = all_cases - len_valid
    else:
        datalist_fname = DATASET_MAPPING[framework][tensor_dim][data_name].get("PATH", "")
        all_data_list = get_items(datalist_fname, format="auto")
        len_valid = math.ceil(len(all_data_list) * split)
        len_train = len(all_data_list) - len_valid
    
    ret = True
    if len_train < n_batch_train:
        print(f"Validation batch size ({n_batch_train}) larger than valid data size ({len_train})!")
        ret = False
    if len_valid < n_batch_valid:
        print(f"Validation batch size ({n_batch_valid}) larger than valid data size ({len_valid})!")
        ret = False
    
    return ret


def check_loss(ctx_params):
    output_nc, loss_fn = ctx_params.get("output_nc"), ctx_params.get("criterion")
    if loss_fn == "CE" and output_nc == 1:
        print("Single output channel should use BCE instead of CE loss.")
        return False
    return True


def check_lr_policy(ctx_params):
    if ctx_params.get("lr_policy") == "plateau" and ctx_params.get("valid_interval") != 1:
        Print("Warning: recommand set valid-interval = 1" "when using ReduceLROnPlateau", color="y")
    return True
