import os
import gc
import sys
import shutil
import yaml
import logging
import time
import json
import torch
import numpy as np
from pathlib import Path
from functools import partial
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace as sn

from medlp.models import get_engine, get_test_engine
from medlp.data_io import DATASET_MAPPING
from medlp.data_io.dataio import get_dataloader
from medlp.configures import config as cfg
from medlp.utilities.enum import Phases
import medlp.utilities.arguments as arguments
from medlp.utilities.utils import LogColorFormatter
from medlp.utilities.click_callbacks import get_unknown_options, get_exp_name, input_cropsize, select_gpu

from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from utils_cw import (
    Print,
    prompt_when,
    print_smi,
    PathlibEncoder,
    confirmation,
    check_dir,
    get_items_from_file,
    split_train_test,
)

import click
from ignite.engine import Events
from ignite.utils import setup_logger
from monai_ex.handlers import TensorboardGraphHandler, SNIP_prune_handler
from monai_ex.engines import SupervisedEvaluator, EnsembleEvaluator


def train_core(cargs, files_train, files_valid):
    """Main train function.

    Args:
        cargs (SimpleNamespace): All arguments from cmd line.
        files_train (list): Train file list.
        files_valid (list): Valid file list.
    """
    Print(f"Get {len(files_train)} training data, {len(files_valid)} validation data", color="g")

    # Save param and datalist
    with open(os.path.join(cargs.experiment_path, "train_files.yml"), "w") as f:
        yaml.dump(files_train, f)
    with open(os.path.join(cargs.experiment_path, "valid_files.yml"), "w") as f:
        yaml.dump(files_valid, f)

    train_loader = get_dataloader(cargs, files_train, phase=Phases.TRAIN)
    valid_loader = get_dataloader(cargs, files_valid, phase=Phases.VALID)

    # Tensorboard Logger
    writer = SummaryWriter(log_dir=os.path.join(cargs.experiment_path, "tensorboard"))
    if not cargs.debug and cargs.symbolic_tb:
        tb_dir = check_dir(os.path.dirname(cargs.experiment_path), "tb")
        target_dir = os.path.join(tb_dir, os.path.basename(cargs.experiment_path))
        if os.path.islink(target_dir):
            os.unlink(target_dir)

        os.symlink(
            os.path.join(cargs.experiment_path, "tensorboard"),
            target_dir,
            target_is_directory=True,
        )

    trainer, net = get_engine(cargs, train_loader, valid_loader, writer=writer)
    # logging_level = logging.DEBUG if cargs.debug else logging.INFO
    # trainer.logger = setup_logger(f"{cargs.tensor_dim}-Trainer", level=logging_level, reset=True)
    # stdout_handler = logging.StreamHandler()
    # stdout_handler.setLevel(logging_level)

    # if cargs.compact_log and not cargs.debug:
    #     logging.StreamHandler.terminator = "\r"
    

    trainer.add_event_handler(
        event_name=Events.EPOCH_STARTED,
        handler=lambda x: print("\n", "-" * 15, os.path.basename(cargs.experiment_path), "-" * 15),
    )

    if cargs.visualize:
        Print("Visualize the architecture to tensorboard", color="g")
        trainer.add_event_handler(
            event_name=Events.ITERATION_COMPLETED(once=1),
            handler=TensorboardGraphHandler(net, writer, lambda x: x["image"]),
        )

    if cargs.snip:
        if cargs.snip_percent == 0.0 or cargs.snip_percent == 1.0:
            Print("Invalid snip_percent. Skip SNIP!", color="y")
        else:
            Print("Begin SNIP pruning", color="g")
            snip_device = torch.device("cuda")
            # snip_device = torch.device("cpu")  #! TMP solution to solve OOM issue
            original_device = torch.device("cuda") if cargs.gpus != "-1" else torch.device("cpu")
            trainer.add_event_handler(
                event_name=Events.ITERATION_STARTED(once=1),
                handler=SNIP_prune_handler(
                    trainer.network,
                    trainer.prepare_batch,
                    trainer.loss_function,
                    cargs.snip_percent,
                    trainer.data_loader,
                    device=original_device,
                    snip_device=snip_device,
                    verbose=cargs.debug,
                    logger_name=trainer.logger.name,
                ),
            )

    trainer.run()


@click.command("train", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@arguments.hidden_auxilary_params
@arguments.common_params
@arguments.solver_params
@arguments.network_params
@click.option("--smi", default=True, callback=print_smi, help="Print GPU usage")
@click.option("--gpus", type=str, callback=select_gpu, help="The ID of active GPU")
@click.option("--experiment-path", type=str, callback=get_exp_name, default="")
@click.option(
    "--confirm",
    callback=partial(
        confirmation,
        output_dir_ctx="experiment_path",
        save_code=(cfg.get_medlp_cfg("mode") == "dev"),
        save_dir=cfg.get_medlp_cfg("external_network_dir"),
    ),
)
@click.pass_context
def train(ctx, **args):
    """Entry of train command."""
    auxilary_params = get_unknown_options(ctx)
    args.update(auxilary_params)
    cargs = sn(**args)

    if len(auxilary_params) > 0:  # dump auxilary params
        with cargs.experiment_path.joinpath("param.list").open("w") as f:
            json.dump(args, f, indent=2, sort_keys=True, cls=PathlibEncoder)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        Print("CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cargs.gpus)

    if "," in cargs.gpus:
        cargs.gpu_ids = list(range(len(list(map(int, cargs.gpus.split(","))))))
    else:
        cargs.gpu_ids = [0]

    # ! dump dataset file
    source_file = DATASET_MAPPING[cargs.framework][cargs.tensor_dim][cargs.data_list].get("SOURCE")
    if source_file and os.path.isfile(source_file):
        shutil.copyfile(source_file, cargs.experiment_path.joinpath(f"{cargs.data_list}.snapshot"))

    # ! Manually specified train&valid datalist
    if os.path.isfile(cargs.train_list) and os.path.isfile(cargs.valid_list):
        files_train = get_items_from_file(cargs.train_list, format="auto")
        files_valid = get_items_from_file(cargs.valid_list, format="auto")
        train_core(cargs, files_train, files_valid)
        return cargs

    data_list = DATASET_MAPPING[cargs.framework][cargs.tensor_dim][cargs.data_list].get("PATH", "")
    test_file = DATASET_MAPPING[cargs.framework][cargs.tensor_dim][cargs.data_list].get("TEST_PATH")

    # ! Synthetic test phase
    if data_list is None:
        train_data_num = 100
        Print("Using synthetic test data...", color="y")
        train_datalist = [
            {"image": f"synthetic_image{i}.nii.gz", "label": f"synthetic_label{i}.nii.gz"}
            for i in range(train_data_num)
        ]
    else:
        assert os.path.isfile(data_list), f"Data list '{data_list}' not exists!"
        train_datalist = get_items_from_file(data_list, format="auto")
        # test_datalist = []

    if cargs.do_test and (test_file is None or not os.path.isfile(test_file)):
        Print(
            "Test datalist is not found, split test cohort from training data with split ratio of {cargs.split}",
            color="y",
        )
        train_test_cohort = split_train_test(
            train_datalist, cargs.split, cfg.get_key("label"), 1, random_seed=cargs.seed
        )
        train_datalist, test_datalist = train_test_cohort[0]

    if cargs.partial < 1:
        Print("Use {} data".format(int(len(train_datalist) * cargs.partial)), color="y")
        train_datalist = train_datalist[: int(len(train_datalist) * cargs.partial)]
    elif cargs.partial > 1:
        Print(f"Expect partial < 1, but got'{cargs.partial}'. Ignored.")

    cargs.split = int(cargs.split) if cargs.split >= 1 else cargs.split
    if cargs.n_fold > 1 or cargs.n_repeat > 1:  # ! K-fold cross-validation
        if cargs.n_fold > 1:
            folds = cargs.n_fold
            kf = KFold(n_splits=cargs.n_fold, random_state=cargs.seed, shuffle=True)
        elif cargs.n_repeat > 1:
            folds = cargs.n_repeat
            kf = ShuffleSplit(n_splits=cargs.n_repeat, test_size=cargs.split, random_state=cargs.seed)
        else:
            raise ValueError(f"Got unexpected n_fold({cargs.n_fold}) or n_repeat({cargs.n_repeat})")

        for i, (train_index, test_index) in enumerate(kf.split(train_datalist)):
            ith = i if cargs.ith_fold < 0 else cargs.ith_fold
            if i < ith:
                continue
            Print(f"Processing {i+1}/{folds} cross-validation", color="g")
            train_data = list(np.array(train_datalist)[train_index])
            valid_data = list(np.array(train_datalist)[test_index])

            if "-th" in os.path.basename(cargs.experiment_path):
                cargs.experiment_path = check_dir(os.path.dirname(cargs.experiment_path), f"{i}-th")
            else:
                cargs.experiment_path = check_dir(cargs.experiment_path, f"{i}-th")

            # copy param.list to i-fold dir
            with cargs.experiment_path.joinpath("param.list").open("w") as f:
                fold_args = args.copy()
                fold_args["n_fold"] = fold_args["n_repeat"] = 0
                fold_args["experiment_path"] = str(cargs.experiment_path)
                json.dump(fold_args, f, indent=2)

            train_core(cargs, train_data, valid_data)
            Print("Cleaning CUDA cache...", color="g")
            gc.collect()
            torch.cuda.empty_cache()
    else:  # ! Plain training
        train_data, valid_data = train_test_split(train_datalist, test_size=cargs.split, random_state=cargs.seed)
        train_core(cargs, train_data, valid_data)

    # ! Do testing
    if cargs.do_test > 0:
        if test_file and os.path.isfile(test_file):
            test_datalist = get_items_from_file(test_file, format="auto")
        elif len(test_datalist) > 0:
            test_file = cargs.experiment_path.joinpath("test_files.yml")
            with test_file.open("w") as f:
                yaml.dump(test_datalist, f)
        else:
            return cargs

        has_labels = np.all([cfg.get_key("label") in item for item in test_datalist])

        if len(test_datalist) > 0:
            configures = {
                "config": os.path.join(args["experiment_path"], "param.list"),
                "test_files": test_file,
                "with_label": has_labels,
                "use_best_model": True,
                "smi": False,
                "gpus": args["gpus"],
            }
            print("****configure:", configures)
            test_cfg(default_map=configures)

    return cargs


@click.command(
    "train-from-cfg",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@click.option("--config", type=click.Path(exists=True))
@click.argument("additional_args", nargs=-1, type=click.UNPROCESSED)
def train_cfg(**args):
    """Entry of train-from-cfg command"""
    if len(args.get("additional_args")) != 0:  # parse additional args
        Print(
            "*** Lr schedule changes do not work yet! Please make a confirmation at last!***\n",
            color="y",
        )

    configures = get_items_from_file(args["config"], format="json")

    configures["smi"] = False
    gpu_id = click.prompt(f"Current GPU id", default=configures["gpus"])
    configures["gpus"] = gpu_id
    configures["config"] = args["config"]

    train(default_map=configures)
    # ctx.invoke(train, **configures)


@click.command(
    "test-from-cfg",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@click.option("--config", type=click.Path(exists=True), default="YourConfigFle")
@click.option("--test-files", type=str, default="", help="External files (json/yaml) for testing")
@click.option("--out-dir", type=str, default=None, help="Optional output dir to save results")
@click.option(  # TODO: automatically decide when using patchdataset
    "--slidingwindow",
    is_flag=True,
    callback=input_cropsize,
    help="Use slidingwindow sampling",
)
@click.option("--with-label", is_flag=True, help="whether test data has label")
@click.option("--save-image", is_flag=True, help="Save the tested image data")
@click.option("--save-label", is_flag=True, help="Save the tested label data (image type)")
@click.option("--save-latent", is_flag=True, help="Save the latent code")
@click.option("--save-prob", is_flag=True, help="Save predicted probablity")
@click.option(
    "--target-layer",
    type=str,
    callback=partial(prompt_when, keyword="save_latent"),
    help="Target layer of saving latent code",
)
@click.option("--use-best-model", is_flag=True, help="Automatically select best model for testing")
@click.option("--smi", default=True, callback=print_smi, help="Print GPU usage")
@click.option("--gpus", prompt="Choose GPUs[eg: 0]", type=str, help="The ID of active GPU")
def test_cfg(**args):
    """Entry of test-from-cfg command.

    Raises:
        ValueError: External test file (.json/.yaml) must be provided for cross-validation exp!
        ValueError: Test file not exist error.
    """
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    configures = get_items_from_file(args["config"], format="json")

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        Print("CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpus"])

    exp_dir = Path(configures.get("experiment_path", os.path.dirname(args["config"])))
    is_crossvalid = configures.get("n_fold", 0) > 1 or configures.get("n_repeat", 0) > 1

    if os.path.isfile(args["test_files"]):
        test_fpath = args["test_files"]
        test_files = get_items_from_file(args["test_files"], format="auto")
    elif is_crossvalid:
        raise ValueError(
            f"{configures['n_fold']} Cross-validation found!" "You must provide external test file (.json/.yaml)."
        )
    else:
        test_fpaths = list(exp_dir.glob("valid_files*"))
        if len(test_fpaths) > 0:
            test_fpath = test_fpaths[0]
            test_files = get_items_from_file(test_fpath, format="auto")
        else:
            raise ValueError(f"Test/Valid file does not exists in {exp_dir}!")

    if args["use_best_model"]:
        model_list = arguments.get_best_trained_models(exp_dir)
        if is_crossvalid:
            configures["n_fold"] = configures["n_repeat"] = 0
            is_crossvalid = False
    elif is_crossvalid:
        model_list = [""]
    else:
        model_list = [arguments.get_trained_models(exp_dir)]

    configures["preload"] = 0.0
    phase = Phases.TEST_IN if args["with_label"] else Phases.TEST_EX
    configures["phase"] = phase
    configures["save_image"] = args["save_image"]
    configures["save_label"] = args["save_label"]
    configures["save_prob"] = args["save_prob"]
    configures["experiment_path"] = exp_dir
    configures["resample"] = True  # ! departure
    configures["slidingwindow"] = args["slidingwindow"]
    configures["save_latent"] = args["save_latent"]
    configures["target_layer"] = args["target_layer"]
    if args.get("crop_size", None):
        configures["crop_size"] = args["crop_size"]
    configures["out_dir"] = (
        check_dir(args["out_dir"]) if args["out_dir"] else check_dir(exp_dir, f'Test@{time.strftime("%m%d_%H%M")}')
    )

    Print(f"{len(test_files)} test files", color="g")
    test_loader = get_dataloader(sn(**configures), test_files, phase=phase)

    for model_path in model_list:
        configures["model_path"] = model_path

        engine = get_test_engine(sn(**configures), test_loader)
        engine.logger = setup_logger(f"{configures['tensor_dim']}-Tester", level=logging.INFO)

        if isinstance(engine, SupervisedEvaluator):
            Print("Begin testing...", color="g")
        elif isinstance(engine, EnsembleEvaluator):
            Print("Begin ensemble testing...", color="g")

        shutil.copyfile(test_fpath, check_dir(configures["out_dir"]) / os.path.basename(test_fpath))
        engine.run()

        is_intra_ensemble = isinstance(model_path, (list, tuple)) and len(model_path) > 1
        if is_intra_ensemble:
            os.rename(
                configures["out_dir"],
                str(configures["out_dir"]) + "-intra-ensemble",
            )
        elif is_crossvalid:
            os.rename(
                configures["out_dir"],
                str(configures["out_dir"]) + "-ensemble",
            )
        else:
            postfix = model_path[0].stem if isinstance(model_path, (list, tuple)) else model_path.stem
            os.rename(configures["out_dir"], str(configures["out_dir"]) + "-" + postfix)


@click.command(
    "train-and-test",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)
@click.pass_context
def train_and_test(ctx, **args):
    args.update({"do_test": 1})
    train_args = train(default_map=args)
