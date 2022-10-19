import gc
import logging
import os
import shutil
import time
from functools import partial
from pathlib import Path
from types import SimpleNamespace as sn

import click
import numpy as np
import torch
from ignite.engine import Events
from monai.engines.evaluator import EnsembleEvaluator, SupervisedEvaluator
from monai_ex.handlers import SNIP_prune_handler
from monai_ex.utils.misc import Print, check_dir
from torch.utils.tensorboard.writer import SummaryWriter

import strix.utilities.arguments as arguments
import strix.utilities.oyaml as yaml
from strix.configures import config as cfg
from strix.data_io.dataio import get_dataloader
from strix.models import get_engine, get_test_engine
from strix.utilities.click import CommandEx, OptionEx
from strix.utilities.click_callbacks import (
    backup_project,
    check_amp,
    check_batchsize,
    check_freeze_api,
    check_loss,
    check_lr_policy,
    confirmation,
    dump_hyperparameters,
    dump_params,
    get_exp_name,
    get_unknown_options,
    input_cropsize,
    parse_project,
    print_smi,
    prompt_when,
    select_gpu,
)
from strix.utilities.enum import Phases, SerialFileFormat
from strix.utilities.generate_cohorts import generate_test_cohort, generate_train_valid_cohorts
from strix.utilities.utils import get_items, setup_logger

option = partial(click.option, cls=OptionEx)
command = partial(click.command, cls=CommandEx)


def train_core(cargs, files_train, files_valid):
    """Main train function.

    Args:
        cargs (SimpleNamespace): All arguments from cmd line.
        files_train (list): Train file list.
        files_valid (list): Valid file list.
    """
    logger = setup_logger(cargs.logger_name)
    logger.info(f"Get {len(files_train)} training data, {len(files_valid)} validation data")

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
    trainer.add_event_handler(
        event_name=Events.EPOCH_STARTED,
        handler=lambda x: print("\n", "-" * 15, os.path.basename(cargs.experiment_path), "-" * 15),
    )

    if cargs.snip:
        if cargs.snip_percent == 0.0 or cargs.snip_percent == 1.0:
            logger.warn("Invalid snip_percent. Skip SNIP!")
        else:
            logger.info("Begin SNIP pruning")
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

    try:
        trainer.run()
    except SystemExit as e:
        print(f"Training exited with {e}!")
    except RuntimeError as e:
        print("Run time error occured!", e)


train_cmd_history = os.path.join(cfg.get_strix_cfg("cache_dir"), ".strix_train_cmd_history")


@command(
    "train",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "prompt_in_default_map": True,
        "default_map": get_items(train_cmd_history, SerialFileFormat.YAML, allow_filenotfound=True),
    },
)
@arguments.hidden_auxilary_params
@arguments.common_params
@arguments.solver_params
@arguments.network_params
@option("--smi", default=True, callback=print_smi, help="Print GPU usage")
@option("--gpus", type=str, callback=select_gpu, help="The ID of active GPU")
@option("--experiment-path", type=str, callback=get_exp_name, default="")
@option(
    "--dump-params",
    hidden=True,
    is_flag=True,
    default=False,
    callback=partial(dump_params, output_path=train_cmd_history),
)
@option(
    "--confirm",
    callback=partial(
        confirmation,
        checklist=[
            check_batchsize,
            check_loss,
            check_lr_policy,
            check_freeze_api,
            check_amp,
            dump_hyperparameters,
            backup_project,
        ],
    ),
)
@click.pass_context
def train(ctx, **args):
    """Entry of train command."""
    auxilary_params = get_unknown_options(ctx)
    args.update(auxilary_params)
    cargs = sn(**args)

    logger_name = f"{cargs.tensor_dim}-Trainer"
    cargs.logger_name = logger_name
    logging_level = logging.DEBUG if cargs.debug else logging.INFO
    log_path = None if cargs.disable_logfile else cargs.experiment_path.joinpath("logs")
    log_terminator = "\r" if cargs.compact_log and not cargs.debug else "\n"
    logger = setup_logger(logger_name, logging_level, filepath=log_path, reset=True, terminator=log_terminator)

    if len(auxilary_params) > 0:  # dump auxilary params
        with cargs.experiment_path.joinpath("param.list").open("w") as f:
            yaml.dump(args, f, sort_keys=True)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.warn(f"CUDA_VISIBLE_DEVICES specified to {os.environ['CUDA_VISIBLE_DEVICES']}, ignoring --gpus flag")
        cargs.gpus = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cargs.gpus)

    if "," in cargs.gpus:
        cargs.gpu_ids = list(range(len(list(map(int, cargs.gpus.split(","))))))
    else:
        cargs.gpu_ids = [0]

    train_valid_cohorts = generate_train_valid_cohorts(**vars(cargs), logger=logger)

    folds = cargs.n_fold or cargs.n_repeat
    for i, ifold in enumerate(train_valid_cohorts):
        ith = i if cargs.ith_fold < 0 else cargs.ith_fold
        if i < ith:
            continue

        if folds:
            logger.info(f"\n\n\t**** Processing {i+1}/{folds} cross-validation ****\n\n")

        _experiment_path = check_dir(cargs.experiment_path, f"{i}-th") if folds else cargs.experiment_path

        # copy param.list to i-fold dir
        with _experiment_path.joinpath("param.list").open("w") as f:
            fold_args = args.copy()
            fold_args["n_fold"] = fold_args["n_repeat"] = 0
            fold_args["experiment_path"] = str(_experiment_path)
            yaml.dump(fold_args, f, sort_keys=True)

        train_data, valid_data = ifold
        train_core(cargs, train_data, valid_data)

        logger.info("Cleaning CUDA cache...")
        gc.collect()
        torch.cuda.empty_cache()

    # ! Do testing
    if cargs.do_test > 0:
        test_datalist = generate_test_cohort(
            cargs.tensor_dim, cargs.framework, cargs.data_list, cargs.do_test, cargs.split, cargs.seed
        )

        if len(test_datalist) == 0:
            logger.warn("No test data is found! Skip test.")
            return cargs

        testlist_fpath = cargs.experiment_path.joinpath("test_files.yml")
        with testlist_fpath.open("w") as f:
            yaml.dump(test_datalist, f)

        has_labels = np.all([cfg.get_key("label") in item for item in test_datalist])

        configures = {
            "config": os.path.join(args["experiment_path"], "param.list"),
            "test_files": testlist_fpath,
            "with_label": has_labels,
            "use_best_model": True,
            "smi": False,
            "gpus": args["gpus"],
        }
        print("****configure:", configures)
        test_cfg(default_map=configures)

    return cargs


@click.command("train-from-cfg", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@option("--config", type=click.Path(exists=True))
@click.argument("additional_args", nargs=-1, type=click.UNPROCESSED)
def train_cfg(**args):
    """Entry of train-from-cfg command"""
    if len(args.get("additional_args")) != 0:  # parse additional args
        Print("*** Lr schedule changes do not work yet! Please make a confirmation at last!***\n", color="y")

    configures = get_items(args["config"], SerialFileFormat.YAML)

    configures["smi"] = False
    gpu_id = click.prompt("Current GPU id", default=configures["gpus"])
    configures["gpus"] = gpu_id
    configures["config"] = args["config"]

    train(default_map=configures, prompt_in_default_map=False)


@click.command("test-from-cfg", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@option("--project", type=click.Path(), callback=parse_project, default=Path.cwd(), help="Project folder path")
@option("--config", type=click.Path(exists=True), default="YourConfigFle")
@option("--test-files", type=str, default="", help="External files (json/yaml) for testing")
@option("--out-dir", type=str, default=None, help="Optional output dir to save results")
@option(  # TODO: automatically decide when using patchdataset
    "--slidingwindow",
    is_flag=True,
    callback=input_cropsize,
    help="Use slidingwindow sampling",
)
@option("--with-label", is_flag=True, help="whether test data has label")
@option("--save-image", is_flag=True, help="Save the tested image data")
@option("--save-label", is_flag=True, help="Save the tested label data (image type)")
@option("--save-latent", is_flag=True, help="Save the latent code")
@option("--save-prob", is_flag=True, help="Save predicted probablity")
@option(
    "--target-layer",
    type=str,
    callback=partial(prompt_when, keyword="save_latent"),
    help="Target layer of saving latent code",
)
@option("--use-best-model", is_flag=True, help="Automatically select best model for testing")
@option("--smi", default=True, callback=print_smi, help="Print GPU usage")
@option("--gpus", prompt="Choose GPUs[eg: 0]", type=str, help="The ID of active GPU")
def test_cfg(**args):
    """Entry of test-from-cfg command.

    Raises:
        ValueError: External test file (.json/.yaml) must be provided for cross-validation exp!
        ValueError: Test file not exist error.
    """
    configures = get_items(args["config"], SerialFileFormat.YAML)

    logger_name = f"{configures['tensor_dim']}-Tester"
    logging_level = logging.DEBUG if configures["debug"] else logging.INFO
    logger = setup_logger(logger_name, level=logging_level, reset=True)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logger.warn("CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpus"])

    exp_dir = Path(configures.get("experiment_path", os.path.dirname(args["config"])))
    is_crossvalid = configures.get("n_fold", 0) > 1 or configures.get("n_repeat", 0) > 1

    if os.path.isfile(args["test_files"]):
        test_fpath = args["test_files"]
        test_files = get_items(args["test_files"])
    elif is_crossvalid:
        raise ValueError(
            f"{configures['n_fold']} Cross-validation found! You must provide external test file (.json/.yaml)."
        )
    else:
        test_fpaths = list(exp_dir.glob("valid_files*"))
        if len(test_fpaths) > 0:
            test_fpath = test_fpaths[0]
            test_files = get_items(test_fpath)
        else:
            raise ValueError(f"Test/Valid file does not exists in {exp_dir}!")

    if args["use_best_model"]:  #! refactor this!
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
    configures["logger_name"] = logger_name
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

    logger.info(f"{len(test_files)} test files")
    test_loader = get_dataloader(sn(**configures), test_files, phase=phase)

    for model_path in model_list:
        configures["model_path"] = model_path

        engine = get_test_engine(sn(**configures), test_loader)

        if isinstance(engine, SupervisedEvaluator):
            logger.info(" ==== Begin testing ====")
        elif isinstance(engine, EnsembleEvaluator):
            logger.info(" ==== Begin ensemble testing ====")

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
