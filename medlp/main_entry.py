import os
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
from medlp.utilities.handlers import SNIP_prune_handler
from medlp.utilities.click_ex import get_unknown_options
from medlp.configures import config as cfg
import medlp.utilities.click_callbacks as clb

from sklearn.model_selection import train_test_split, KFold, ShuffleSplit
from utils_cw import (
    Print,
    print_smi,
    confirmation,
    check_dir,
    get_items_from_file,
)

import click
from ignite.engine import Events
from ignite.utils import setup_logger
from monai_ex.handlers import CheckpointLoaderEx, TensorboardGraphHandler
from monai_ex.engines import SupervisedEvaluator, EnsembleEvaluator


def train_core(cargs, files_train, files_valid):
    """Main train function.

    Args:
        cargs (SimpleNamespace): All arguments from cmd line.
        files_train (list): Train file list.
        files_valid (list): Valid file list.
    """
    Print(
        f"Get {len(files_train)} training data, {len(files_valid)} validation data",
        color="g",
    )
    # Save param and datalist
    with open(os.path.join(cargs.experiment_path, "train_files.yml"), "w") as f:
        yaml.dump(files_train, f)
    with open(os.path.join(cargs.experiment_path, "valid_files.yml"), "w") as f:
        yaml.dump(files_valid, f)

    train_loader = get_dataloader(cargs, files_train, phase="train")
    valid_loader = get_dataloader(cargs, files_valid, phase="valid")

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

    logging_level = logging.DEBUG if cargs.debug else logging.INFO
    trainer.logger = setup_logger(f"{cargs.tensor_dim}-Trainer", level=logging_level)
    if cargs.compact_log and not cargs.debug:
        logging.StreamHandler.terminator = "\r"

    trainer.add_event_handler(
        event_name=Events.EPOCH_STARTED, handler=lambda x: print("\n", "-" * 40)
    )

    if os.path.isfile(cargs.pretrained_model_path):
        Print(
            "Load pretrained model for contiune training:\n\t",
            cargs.pretrained_model_path,
            color="g",
        )
        trainer.add_event_handler(
            event_name=Events.STARTED,
            handler=CheckpointLoaderEx(
                load_path=cargs.pretrained_model_path,
                load_dict={"net": net},
                strict=False,
                skip_mismatch=True,
            ),
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
            original_device = (
                torch.device("cuda") if cargs.gpus != "-1" else torch.device("cpu")
            )
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
@click.option("--debug", is_flag=True)
@clb.latent_auxilary_params
@clb.common_params
@clb.solver_params
@clb.network_params
@click.option("--smi", default=True, callback=print_smi, help="Print GPU usage")
@click.option("--gpus", prompt="Choose GPUs[eg: 0]", type=str, help="The ID of active GPU")
@click.option("--experiment-path", type=str, callback=clb.get_exp_name, default="")
@click.option(
    "--confirm", callback=partial(
        confirmation,
        output_dir_ctx="experiment_path",
        save_code=(cfg.get_medlp_cfg('mode') == 'dev'),
        exist_ok=False
    )
)
@click.pass_context
def train(ctx, **args):
    """Entry of train command."""
    auxilary_params = get_unknown_options(ctx)
    args.update(auxilary_params)
    cargs = sn(**args)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        Print("CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cargs.gpus)

    cargs.gpu_ids = list(range(len(list(map(int, cargs.gpus.split(","))))))

    if os.path.isfile(cargs.train_list) and os.path.isfile(cargs.valid_list):
        files_train = get_items_from_file(cargs.train_list, format="auto")
        files_valid = get_items_from_file(cargs.valid_list, format="auto")
        train_core(cargs, files_train, files_valid)
        os.sys.exit()

    data_list = DATASET_MAPPING[cargs.framework][cargs.tensor_dim][cargs.data_list + "_fpath"]
    assert os.path.isfile(data_list), "Data list not exists!"
    files_list = get_items_from_file(data_list, format="auto")

    if cargs.partial < 1:
        Print("Use {} data".format(int(len(files_list) * cargs.partial)), color="y")
        files_list = files_list[: int(len(files_list) * cargs.partial)]

    cargs.split = int(cargs.split) if cargs.split >= 1 else cargs.split
    if cargs.n_fold > 1 or cargs.n_repeat > 1:  #! K-fold cross-validation
        if cargs.n_fold > 1:
            folds = cargs.n_fold
            kf = KFold(n_splits=cargs.n_fold, random_state=cargs.seed, shuffle=True)
        elif cargs.n_repeat > 1:
            folds = cargs.n_repeat
            kf = ShuffleSplit(n_splits=cargs.n_repeat, test_size=cargs.split, random_state=cargs.seed)
        else:
            raise ValueError(f'Got unexpected n_fold({cargs.n_fold}) or n_repeat({cargs.n_repeat})')

        for i, (train_index, test_index) in enumerate(kf.split(files_list)):
            ith = i if cargs.ith_fold < 0 else cargs.ith_fold
            if i < ith:
                continue
            Print(f"Processing {i+1}/{folds} cross-validation", color="g")
            files_train = list(np.array(files_list)[train_index])
            files_valid = list(np.array(files_list)[test_index])

            if "-th" in os.path.basename(cargs.experiment_path):
                cargs.experiment_path = check_dir(os.path.dirname(cargs.experiment_path), f"{i}-th")
            else:
                cargs.experiment_path = check_dir(cargs.experiment_path, f"{i}-th")

            # copy param.list to fold dir
            with cargs.experiment_path.joinpath('param.list').open('w') as f:
                args['n_fold'] = args['n_repeat'] = 0
                args['experiment_path'] = str(cargs.experiment_path)
                json.dump(args, f, indent=2)

            train_core(cargs, files_train, files_valid)
            Print("Cleaning CUDA cache...", color="g")
            torch.cuda.empty_cache()        
    else:  #! Plain training
        files_train, files_valid = train_test_split(
            files_list, test_size=cargs.split, random_state=cargs.seed
        )
        train_core(cargs, files_train, files_valid)


@click.command("train-from-cfg", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
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
    # click.confirm(f"Loading configures: {configures}", default=True, abort=True, show_default=True)

    configures["smi"] = False
    gpu_id = click.prompt(f"Current GPU id: {configures['gpus']}")
    configures["gpus"] = gpu_id
    configures["config"] = args["config"]

    train(default_map=configures)
    # ctx.invoke(train, **configures)


@click.command("test-from-cfg")
@click.option("--config", type=click.Path(exists=True))
@click.option("--test-files", type=str, default="", help="External files (.json) for testing")
@click.option("--out-dir", type=str, default=None, help="Optional output dir to save results")
@click.option(  # TODO: automatically decide when using patchdataset
    "--slidingwindow", is_flag=True, callback=clb.input_cropsize, help='Use slidingwindow sampling'
)
@click.option("--with-label", is_flag=True, help="whether test data has label")
@click.option("--save-image", is_flag=True, help="Save the tested image data")
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
    if os.path.isfile(args["test_files"]):
        test_fpath = args["test_files"]
        test_files = get_items_from_file(args["test_files"], format="auto")
    elif configures.get("n_fold", 0) > 1:
        raise ValueError(
            f"{configures['n_fold']} Cross-validation found!"
            "You must provide external test file (.json/.yaml)."
        )
    else:
        test_fpaths = list(exp_dir.glob('test_files*'))
        if len(test_fpaths) > 0:
            test_fpath = test_fpaths[0]
            test_files = get_items_from_file(test_fpath, format="auto")
        else:
            raise ValueError(f"Test file does not exists in {exp_dir}!")

    # configures["model_path"] = (
    #     clb.get_trained_models(exp_dir, args['use_best_model']) if configures.get("n_fold", 0) <= 1 else None
    # )
    best_models = clb.get_trained_models(exp_dir, args['use_best_model']) if \
                configures.get("n_fold", 0) <= 1 or configures.get("n_repeat", 0) <= 1 else [None]

    configures["preload"] = 0.0
    phase = "test" if args["with_label"] else "test_wo_label"
    configures["phase"] = phase
    configures["save_image"] = args["save_image"]
    configures["experiment_path"] = exp_dir
    configures['resample'] = True  #! departure
    configures['slidingwindow'] = args['slidingwindow']
    if args.get('crop_size', None):
        configures['crop_size'] = args['crop_size']
    configures["out_dir"] = (
        check_dir(args["out_dir"])
        if args["out_dir"]
        else check_dir(exp_dir, f'Test@{time.strftime("%m%d_%H%M")}')
    )

    Print(f"{len(test_files)} test files", color="g")
    test_loader = get_dataloader(sn(**configures), test_files, phase=phase)

    for model_path in best_models:
        configures['model_path'] = str(model_path)

        engine = get_test_engine(sn(**configures), test_loader)
        engine.logger = setup_logger(
            f"{configures['tensor_dim']}-Tester", level=logging.INFO
        )  #! not work

        if isinstance(engine, SupervisedEvaluator):
            Print("Begin testing...", color="g")
        elif isinstance(engine, EnsembleEvaluator):
            Print("Begin ensemble testing...", color="g")

        shutil.copyfile(
            test_fpath, check_dir(configures["out_dir"])/os.path.basename(test_fpath)
        )
        engine.run()
        os.rename(configures["out_dir"], str(configures["out_dir"])+"-"+model_path.stem)


@click.command("unlink")
@click.option(
    "--root-dir", type=click.Path(exists=True), help="Root dir contains symbolic dirs"
)
@click.option(
    "-a",
    "--all-dir",
    is_flag=True,
    help="Unlink all dirs including both avalible and unavailable dirs",
)
def unlink_dirs(root_dir, all_dir):
    """Utility for unlink invalid symbolic tensorboard dir.

    Args:
        root_dir (str): Root dir contains symbolic dirs.
        all_dir (bool): whether unlink both invalid and valid sym dirs.
    """
    for d in os.listdir(root_dir):
        d = os.path.join(root_dir, d)
        if os.path.islink(d):
            if not os.path.isdir(d):
                os.unlink(d)
                print("Unlinked unavailable symbolic dir:", d)
            elif all_dir:
                os.unlink(d)
                print("Unlinked symbolic dir:", d)
            else:
                pass
