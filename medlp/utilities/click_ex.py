import inspect
from functools import partial
from pathlib import Path
from time import strftime
from types import SimpleNamespace as sn

from click import Choice, ParamType, prompt
from click.types import convert_type
from medlp.data_io import DATASET_MAPPING
from medlp.models import ARCHI_MAPPING
from medlp.models.cnn.losses import LOSS_MAPPING
from medlp.utilities.enum import BUILTIN_TYPES, FRAMEWORKS, Frameworks
from medlp.utilities.utils import is_avaible_size
from termcolor import colored
from utils_cw import Print, check_dir, get_items_from_file


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
        if isinstance(value, str) and value.isnumeric():
            value = int(value)

        if value in self.choicemap:
            return self.choicemap[value]
        elif value in self.choicemap.values():
            return value
        else:
            self.fail(
                f"invaid index choice: {value}. Please input integer index or correct value!"
            )


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
    partial_data = "-partial" if "partial" in ctx.params and ctx.params["partial"] < 1 else ""

    if ctx.params["lr_policy"] == "plateau" and ctx.params["valid_interval"] != 1:
        Print("Warning: recommand set valid-interval = 1" "when using ReduceLROnPlateau", color="y")

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
    if ctx.params["framework"] == Frameworks.MULTITASK.value:
        loss_fn = '+'.join(ctx.params['criterion']) 
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

    if project_name:
        proj_dirname = f"Project-{project_name}"
        return (Path(ctx.params["out_dir"]) / ctx.params["framework"] / proj_dirname / datalist_name / exp_name)
    else:
        return (Path(ctx.params["out_dir"]) / ctx.params["framework"] / datalist_name / exp_name)


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
        steps = _prompt("steps", tuple, (100, 300), value_proc=lambda x: list(map(int, x.split(","))))
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
    if value == Frameworks.MULTITASK.value and not ctx.params.get('subtask1'):
        if "multitask" in FRAMEWORKS:
            FRAMEWORKS.remove("multitask")
        subtask1 = prompt(f"Sub {colored('task1', 'yellow')}", type=NumericChoice(FRAMEWORKS), default=2)
        subtask2 = prompt(f"Sub {colored('task2', 'yellow')}", type=NumericChoice(FRAMEWORKS), default=1)
        ctx.params['subtask1'] = subtask1
        ctx.params['subtask2'] = subtask2
    return value


def loss_select(ctx, param, value, prompt_all_args=False):

    def _single_loss_select(ctx, framework, value, prompt_all_args, meta_key_postfix=''):
        losslist = list(LOSS_MAPPING[framework].keys())

        assert len(losslist) > 0, f"No loss available for {framework}! Abort!"
        if value is not None and value in losslist:
            return value

        prompts = f"Loss_fn for {colored(meta_key_postfix[1:], 'yellow')}" if meta_key_postfix else "Loss_fn"
        value = prompt(prompts, type=NumericChoice(losslist))

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
                    loss_params[k] = _prompt(
                        f"Loss argument '{k}'", anno[k], default_value
                    )
                else:
                    print(f"Cannot handle type '{anno.get(k)}' for argment '{k}'")
                    # raise ValueError(f"Cannot handle type '{anno.get(k)}' for argment '{k}'")
            ctx.params[meta_key] = loss_params

        return value

    if ctx.params["framework"] == Frameworks.MULTITASK.value:
        loss1 = _single_loss_select(
            ctx, ctx.params["subtask1"], value, prompt_all_args, '_task1'
        )
        loss2 = _single_loss_select(
            ctx, ctx.params["subtask2"], value, prompt_all_args, '_task2'
        )
        return [loss1, loss2]
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
