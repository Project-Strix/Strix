import os
import sys
import logging
from functools import partial
from monai_ex.transforms import *
from monai_ex.data import *
from monai_ex.utils import ensure_tuple, first
import medlp.utilities.oyaml as yaml
from medlp.utilities.enum import DIMS, FRAMEWORK_TYPES, PHASES
from medlp.data_io.dataio import DATASET_MAPPING
from utils_cw import get_items_from_file

root_tree = {
    "ATTRIBUTE": {
        "DIM": "",
        "NAME": "",
        "FRAMEWORK": "",
        "KEYS": ("image", "label"),
        "PHASE": ("train", "valid", "test"),
        "FILES_LIST": "",
    },
    "PREPROCESS": {
        "LOADER": {},
        "CHANNELER": {},
        "ORIENTER": {},
        "RESCALER": {},
        "RESIZER": {},
        "CROPADER": {},
    },
    "AUGMENTATION": {},
    "DATASET_TYPE": {},
    "DATALOADER": {},
}

mapping = {
    "LOADER": LOADER,
    "CHANNELER": CHANNELER,
    "ORIENTER": ORIENTER,
    "RESCALER": RESCALER,
    "RESIZER": RESIZER,
    "CROPADER": CROPADER,
    "AUGMENTOR": AUGMENTOR,
    "UTILS": UTILS,
    "DATASET_TYPE": DATASETYPE,
}


def check_config(configs, key, candidates=None):
    cfg = configs.copy()
    for k in key:
        try:
            cfg = cfg[k]
        except:
            raise ValueError("Missing key: {} in dict {}".format(k, key))

    if candidates is not None:
        for item in ensure_tuple(cfg):
            assert (
                item in candidates
            ), f"Key '{item}' in '{key[-1]}' is not in '{candidates}'"


def parse_dataset_config(configs):
    # necessary keys
    check_config(configs, key=["ATTRIBUTE", "DIM"], candidates=DIMS)
    check_config(configs, key=["ATTRIBUTE", "NAME"])
    check_config(configs, key=["ATTRIBUTE", "FRAMEWORK"], candidates=FRAMEWORK_TYPES)
    check_config(configs, key=["ATTRIBUTE", "KEYS"])
    check_config(
        configs, key=["ATTRIBUTE", "PHASE"], candidates=["train", "valid", "test"]
    )
    check_config(configs, key=["ATTRIBUTE", "FILES_LIST"])
    check_config(configs, key=["PREPROCESS", "LOADER"])

    file_list = configs["ATTRIBUTE"]["FILES_LIST"]
    assert os.path.isfile(file_list), f"File list not exist! {file_list}"

    default_keys = configs["ATTRIBUTE"]["KEYS"]
    default_phase = configs["ATTRIBUTE"]["PHASE"]

    #! Preprocess
    preprocessors = {"train": [], "valid": [], "test": []}
    for processor in configs["PREPROCESS"]:
        for step in configs["PREPROCESS"][processor]:
            arguments = {
                "keys": configs["PREPROCESS"][processor][step].get("keys", default_keys)
            }
            args = configs["PREPROCESS"][processor][step].get("args", {})
            phases = configs["PREPROCESS"][processor][step].get("phase", default_phase)
            phases = ensure_tuple(phases)

            assert isinstance(args, dict),\
                f"args must be dict, but got {args} ({type(args)})"
            arguments.update(args)

            func = mapping[processor][step]

            try:  # TODO extract to function?
                fn = func(**arguments)
            except TypeError as e:
                print(
                    f"Input argument error!\n"
                    f"{step} got unexpected keyword argument: {args}\n"
                    f"Msg: {e}"
                )
                sys.exit(1)
            except Exception as e:
                print(f"Error msg: {e}")
                sys.exit(1)
            else:
                for p in phases:
                    preprocessors[p].append(fn)

    #! Augmentation
    augmentations = {"train": [], "valid": [], "test": []}
    if configs.get("AUGMENTATION", None):
        for processor in configs["AUGMENTATION"]:
            arguments = {
                "keys": configs["AUGMENTATION"][processor].get("keys", default_keys)
            }
            args = configs["AUGMENTATION"][processor].get("args", {})
            phases = configs["AUGMENTATION"][processor].get(
                "phase", "train"
            )  # default only in train phase
            phases = ensure_tuple(phases)

            assert isinstance(args, dict),\
                f"args must be dict, but got {args} ({type(args)})"
            arguments.update(args)

            func = mapping["AUGMENTOR"][processor]

            try:
                fn = func(**arguments)
            except TypeError as e:
                print(
                    f"Input argument error!\n"
                    f"{processor} got unexpected keyword argument: {args} \n"
                    f"Msg: {e}"
                )
                sys.exit(1)
            except ValueError as e:
                print(
                    f"Input argument error!\n"
                    f"{processor} got incompatible arguments.\n"
                    f"Msg: {e}"
                )
                sys.exit(1)
            except Exception as e:
                print(f"Error msg: {e}")
                sys.exit(1)
            else:
                for p in augmentations:
                    augmentations[p].append(fn)

    transforms = {
        phase: ComposeEx(preprocessors.get(phase, []) + augmentations.get(phase, []))
        for phase in PHASES
    }

    #! Dataset type
    if configs.get("DATASET_TYPE", None):
        dataset_name = list(configs["DATASET_TYPE"].keys())
        assert (  # TODO: support multi?
            len(dataset_name) == 1
        ), "Currently we only support sinlge dataset type for both train&valid"
        phases = configs["DATASET_TYPE"][dataset_name[0]].get(
            "phase", ["train", "valid"]
        )
        args = configs["DATASET_TYPE"][dataset_name[0]].get("args", {})

        datasets = {phase: [DATASETYPE[dataset_name[0]], args] for phase in phases}
    else:
        datasets = None

    #! DataLoader
    if configs.get("DATALOADER", None):
        raise NotImplementedError
    else:
        dataloader = None

    return datasets, dataloader, transforms


def create_dataset_from_cfg(
    files_list,
    phase,
    opts,
    datasets,
    dataloader,
    transforms
):
    if not datasets:
        # use default dataset
        tensor_dim = opts.get("tensor_dim", None)
        if tensor_dim == "2D":
            dataset_ = DATASETYPE["CacheDataset"]
            args_ = {"cache_rate": 0.0 if phase is "test" else opts.get("preload", 1)}
        elif tensor_dim == "3D":
            dataset_ = DATASETYPE["PersistentDataset"]
            args_ = {"cache_dir": opts.get("cache_dir", "./")}
        else:
            raise ValueError(
                f"Invalid tensor dim '{tensor_dim}'"
                f"Please make sure you have set tensor_dim in cmd line"
                f"or setup in cfg file."
            )
    else:
        # use custom dataset
        dataset_ = datasets[phase][0]
        args_ = datasets[phase][1]

    args = {"data": files_list, "transform": transforms[phase]}
    args.update(args_)

    if dataloader is None:
        return dataset_(**args)
    else:
        raise NotImplementedError


def register_dataset_from_cfg(config_path):
    assert config_path.is_file(), f"Config file is not found: {config_path}"
    with config_path.open() as f:
        configs = yaml.full_load(f)

    datasets_, dataloader_, transforms_ = parse_dataset_config(configs)

    DATASET_MAPPING[configs["ATTRIBUTE"]["FRAMEWORK"]].register(
        configs["ATTRIBUTE"]["DIM"],
        configs["ATTRIBUTE"]["NAME"],
        configs["ATTRIBUTE"]["FILES_LIST"],
        partial(
            create_dataset_from_cfg,
            datasets=datasets_,
            dataloader=dataloader_,
            transforms=transforms_,
        ),
    )

def test_dataset_from_config(config_path, phase, opts):
    assert config_path.is_file(), f"Config file is not found: {config_path}"
    with config_path.open() as f:
        configs = yaml.full_load(f)

    datasets_, dataloader_, transforms_ = parse_dataset_config(configs)
    transforms_[phase].add_transforms(
        DataStatsD(
            keys=configs["ATTRIBUTE"]["KEYS"],
            prefix=configs["ATTRIBUTE"]["KEYS"],
            logger_handler=logging.StreamHandler(sys.stdout) #! No output!
            )
    )
    dataset = create_dataset_from_cfg(
        get_items_from_file(configs["ATTRIBUTE"]["FILES_LIST"], format='auto'),
        phase,
        opts,
        datasets_,
        dataloader_,
        transforms_
    )
    first(dataset)


if __name__ == "__main__":
    from pathlib import Path

    register_dataset_from_cfg(Path(r"/homes/clwang/test_config.yaml"))
