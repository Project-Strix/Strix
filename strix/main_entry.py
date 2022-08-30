import gc
import shutil
import logging
import os
import shutil
import time
import torch
import numpy as np
from pathlib import Path
from functools import partial
from pathlib import Path
from types import SimpleNamespace as sn

import click
import numpy as np
import torch
import yaml
from ignite.engine import Events
from monai_ex.engines import EnsembleEvaluator, SupervisedEvaluator
from monai_ex.handlers import SNIP_prune_handler
from sklearn.model_selection import KFold, ShuffleSplit, train_test_split
from torch.utils.tensorboard import SummaryWriter
from utils_cw import Print, check_dir, split_train_test

import strix.utilities.arguments as arguments
from strix.configures import config as cfg
from strix.models import get_engine, get_test_engine
from strix.utilities.registry import DatasetRegistry
from strix.data_io.dataio import get_dataloader
from strix.configures import config as cfg
from strix.utilities.enum import Phases, Frameworks
from strix.utilities.click import OptionEx, CommandEx
import strix.utilities.oyaml as yaml
import strix.utilities.arguments as arguments
from strix.utilities.utils import setup_logger, get_items, parse_datalist, generate_synthetic_datalist
from strix.utilities.click_callbacks import (
    check_batchsize,
    check_freeze_api,
    check_loss,
    check_lr_policy,
    dump_params,
    confirmation,
    print_smi,
    prompt_when,
    check_amp,
    dump_hyperparameters,
    backup_project,
    select_gpu,
    get_exp_name,
    input_cropsize,
    get_unknown_options,
)


option = partial(click.option, cls=OptionEx)
command = partial(click.command, cls=CommandEx)

def train_core(cargs, train_files, valid_files, unlabel_files=None):
    """Main train function.

    Args:
        cargs (SimpleNamespace): All arguments from cmd line.
        train_files (list): Train file list.
        valid_files (list): Valid file list.
        unlabel_files (list, optional): Unlabeled file list.
    """
    logger = setup_logger(cargs.logger_name)
    logger.info(f"Get {len(train_files)} training data, {len(valid_files)} validation data")

    # Save datalist
    with open(os.path.join(cargs.experiment_path, "train_files.yml"), "w") as f:
        yaml.dump(train_files, f)
    with open(os.path.join(cargs.experiment_path, "valid_files.yml"), "w") as f:
        yaml.dump(valid_files, f)
    if unlabel_files:
        with open(os.path.join(cargs.experiment_path, "unlabel_files.yml"), "w") as f:
            yaml.dump(unlabel_files, f)

    train_loader = get_dataloader(cargs, train_files, phase=Phases.TRAIN)
    valid_loader = get_dataloader(cargs, valid_files, phase=Phases.VALID)
    unlabel_loader = None
    if unlabel_files:
        unlabel_loader = get_dataloader(cargs, unlabel_files, phase=Phases.TRAIN, is_unlabel=True)

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

    trainer, net = get_engine(cargs, train_loader, valid_loader, unlabel_loader=unlabel_loader, writer=writer)
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
        "default_map": get_items(train_cmd_history, format="yaml", allow_filenotfound=True),
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

    # ! if is semi-supervised learning
    is_semisupervise = cargs.semi_supervised or cargs.framework == Frameworks.SEMISUPERVISED.value
    unlabel_files = None

    # ! dump dataset file
    datasets = DatasetRegistry()
    strix_dataset = datasets.get(cargs.tensor_dim, cargs.framework, cargs.data_list)
    if strix_dataset is None:
        raise ValueError(f"Dataset {cargs.data_list} not found!")

    source_file = strix_dataset.get("SOURCE")
    if source_file and os.path.isfile(source_file):
        shutil.copyfile(source_file, cargs.experiment_path.joinpath(f"{cargs.data_list}.snapshot"))

    # ! Manually specified train&valid datalist
    if cargs.train_list and cargs.valid_list:
        if is_semisupervise:
            train_files, unlabel_files = parse_datalist(cargs.train_list, format="auto", has_unlabel=True)
        else:
            train_files = parse_datalist(cargs.train_list, format="auto")
        valid_files = parse_datalist(cargs.valid_list, format="auto")  # valid dataset need no unlabel

        train_core(cargs, train_files, valid_files, unlabel_files)
        return cargs

    datalist_fpath = strix_dataset.get("PATH", "")
    testlist_fpath = strix_dataset.get("TEST_PATH")

    # ! Synthetic test phase
    if datalist_fpath is None:
        train_datalist = generate_synthetic_datalist(100, logger)
    else:
        assert os.path.isfile(datalist_fpath), f"Data list '{datalist_fpath}' not exists!"

        if is_semisupervise:
            train_files, unlabel_files = parse_datalist(datalist_fpath, format="auto", has_unlabel=True)
        else:
            train_files = parse_datalist(datalist_fpath, format="auto")


    if cargs.do_test and (testlist_fpath is None or not os.path.isfile(testlist_fpath)):
        logger.warn(
            f"Test datalist is not found, split test cohort from training data with split ratio of {cargs.split}"
        )
        train_test_cohort = split_train_test(
            train_files, cargs.split, cfg.get_key("label"), 1, random_seed=cargs.seed
        )
        train_files, test_files = train_test_cohort[0]

    if 0 < cargs.partial < 1:
        logger.info("Use {} data".format(int(len(train_files) * cargs.partial)))
        train_files = train_files[: int(len(train_files) * cargs.partial)]
    elif cargs.partial > 1 or cargs.partial == 0:
        logger.warn(f"Expect 0 < partial < 1, but got {cargs.partial}. Ignored.")

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

        for i, (train_index, test_index) in enumerate(kf.split(train_files)):
            ith = i if cargs.ith_fold < 0 else cargs.ith_fold
            if i < ith:
                continue
            logger.info(f"\n\n\t**** Processing {i+1}/{folds} cross-validation ****\n\n")
            train_data = list(np.array(train_files)[train_index])
            valid_data = list(np.array(train_files)[test_index])

            if "-th" in os.path.basename(cargs.experiment_path):
                cargs.experiment_path = check_dir(os.path.dirname(cargs.experiment_path), f"{i}-th")
            else:
                cargs.experiment_path = check_dir(cargs.experiment_path, f"{i}-th")

            # copy param.list to i-fold dir
            with cargs.experiment_path.joinpath("param.list").open("w") as f:
                fold_args = args.copy()
                fold_args["n_fold"] = fold_args["n_repeat"] = 0
                fold_args["experiment_path"] = str(cargs.experiment_path)
                yaml.dump(fold_args, f, sort_keys=True)

            train_core(cargs, train_data, valid_data, unlabel_files)
            logger.info("Cleaning CUDA cache...")
            gc.collect()
            torch.cuda.empty_cache()
    else:  # ! Plain training
        train_data, valid_data = train_test_split(train_files, test_size=cargs.split, random_state=cargs.seed)
        train_core(cargs, train_data, valid_data, unlabel_files)

    # ! Do testing
    if cargs.do_test > 0:
        if testlist_fpath and os.path.isfile(testlist_fpath):
            test_datalist = parse_datalist(testlist_fpath, format="auto")
        elif len(test_datalist) > 0:
            testlist_fpath = cargs.experiment_path.joinpath("test_files.yml")
            with testlist_fpath.open("w") as f:
                yaml.dump(test_datalist, f)
        else:
            return cargs

        has_labels = np.all([cfg.get_key("label") in item for item in test_files])

        if len(test_files) > 0:
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

    configures = get_items(args["config"], format="yaml")

    configures["smi"] = False
    gpu_id = click.prompt("Current GPU id", default=configures["gpus"])
    configures["gpus"] = gpu_id
    configures["config"] = args["config"]

    train(default_map=configures, prompt_in_default_map=False)


@click.command("test-from-cfg", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
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
    configures = get_items(args["config"], format="yaml")

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
        test_files = parse_datalist(args["test_files"], format="auto")
    elif is_crossvalid:
        raise ValueError(
            f"{configures['n_fold']} Cross-validation found! You must provide external test file (.json/.yaml)."
        )
    else:
        test_fpaths = list(exp_dir.glob("valid_files*"))
        if len(test_fpaths) > 0:
            test_fpath = test_fpaths[0]
            test_files = parse_datalist(test_fpath, format="auto")
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
