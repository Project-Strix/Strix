import os
import logging
from time import strftime
from pathlib import Path
from functools import partial
from types import SimpleNamespace as sn

import click
import torch
from utils_cw import get_items_from_file, Print, check_dir
from ignite.engine import Events
from ignite.utils import setup_logger

from strix.models import get_test_engine
from strix.data_io.dataio import get_dataloader
from strix.utilities.utils import get_specify_file
from strix.utilities.enum import Phases
from strix.utilities.click_callbacks import parse_input_str
from strix.utilities.click import NumericChoice as Choice
from strix.utilities.arguments import get_trained_models
from monai_ex.handlers import GradCamHandler


@click.command("gradcam-from-cfg")
@click.option(
    "--config", type=click.Path(exists=True), default="", help="Specify config file"
)
@click.option(
    "--test-files", type=str, default="", help="External files (json/yaml) for testing"
)
@click.option(
    "--target-layer",
    "-L",
    type=click.UNPROCESSED,
    prompt=True,
    callback=partial(parse_input_str, dtype=str),
    help="Target network layer for cam visualization",
)
@click.option(
    "--target-class",
    type=int,
    prompt=True,
    default=1,
    help="GradCAM target class, should be 0 for single chn output",
)
@click.option(
    "--method",
    type=Choice(["gradcam", "layercam"]),
    default="gradcam",
    help="Choose visualize method",
)
@click.option(
    "--out-dir", type=str, default=None, help="Optional output dir to save results"
)
@click.option("--fusion", is_flag=True, help="Normal fusion of multiple CAM maps")
@click.option(
    "--hierarchical", is_flag=True, help="Hierarchically fuse muliple CAM maps"
)
@click.option("--debug", is_flag=True, help="Debug mode. In debug mode, n_sample=1")
@click.option(
    "--n-sample",
    type=int,
    default=-1,
    help="Number of samples to be processed. Defulat=-1",
)
@click.option("--gpus", prompt="Choose GPUs[eg: 0]", type=str)
def gradcam(**args):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        Print("CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpus"])

    configures = get_items_from_file(args["config"], format="json")
    exp_dir = Path(configures.get("experiment_path", os.path.dirname(args["config"])))

    if configures.get("n_fold", 0) > 1:
        raise NotImplementedError("Donot support cross-valid experiment")

    if os.path.isfile(args["test_files"]):
        test_fpath = args["test_files"]
        test_files = get_items_from_file(args["test_files"], format="auto")
    elif get_specify_file(exp_dir, "test_files*"):
        test_fpath = get_specify_file(exp_dir, "test_files*")
        test_files = get_items_from_file(test_fpath, format="auto")
    elif get_specify_file(exp_dir, "valid_files*"):
        test_fpath = get_specify_file(exp_dir, "valid_files*")
        test_files = get_items_from_file(test_fpath, format="auto")
    else:
        raise ValueError(f"Test file does not exists in {exp_dir}!")

    Print(f"Used test file: {test_fpath}", color="green")

    if args["debug"]:
        test_files = test_files[:1]
    elif args["n_sample"] > 0:
        test_files = test_files[: args["n_sample"]]

    phase = Phases.TEST_EX
    configures["preload"] = 0.0
    configures["phase"] = phase
    configures["experiment_path"] = exp_dir
    configures["model_path"] = get_trained_models(exp_dir)[0]  # Get the first model
    configures["save_results"] = False
    configures["save_latent"] = False
    configures["target_layer"] = None
    configures["out_dir"] = (
        check_dir(args["out_dir"])
        if args["out_dir"]
        else check_dir(exp_dir, f"{args['method']}@{strftime('%m%d_%H%M')}")
    )

    Print(f"{len(test_files)} test files", color="g")
    test_dataloader = get_dataloader(sn(**configures), test_files, phase=phase)
    engine = get_test_engine(sn(**configures), test_dataloader)

    logging_level = logging.DEBUG if args["debug"] else logging.INFO
    engine.logger = setup_logger(f"{args['method']}-interpreter", level=logging_level)

    if args["fusion"] and args["hierarchical"]:
        raise ValueError("fusion and hierarchical cannot be both True")

    if args["hierarchical"] and args["method"] == "gradcam":
        engine.logger.warn("Gradcam do not support hierarchical fusion")

    engine.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(once=1),
        handler=GradCamHandler(
            engine.network,
            args["target_layer"],
            args["target_class"],
            engine.data_loader,
            engine.prepare_batch,
            method=args["method"],
            fusion=args["fusion"],
            hierarchical=args["hierarchical"],
            save_dir=configures["out_dir"],
            device=torch.device("cuda")
            if args["gpus"] != "-1"
            else torch.device("cpu"),
            logger_name=engine.logger.name,
        ),
    )

    engine.run()
