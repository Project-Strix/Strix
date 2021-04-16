import os
import time
from pathlib import Path
from types import SimpleNamespace as sn

import click
import torch
from utils_cw import get_items_from_file, Print, check_dir
from ignite.engine import Events

from medlp.models import get_test_engine
from medlp.data_io.dataio import get_dataloader
from medlp.utilities.click_ex import NumericChoice as Choice
from medlp.utilities.enum import FRAMEWORK_TYPES, OUTPUT_DIR
from medlp.utilities.click_callbacks import get_trained_models
from medlp.utilities.handlers import GradCamHandler


@click.command('gradcam-from-cfg')
@click.option("--config", type=click.Path(exists=True))
@click.option("--test-files", type=str, default="", help="External files (json/yaml) for testing")
@click.option('--target-layer', type=str, prompt=True, )
@click.option('--target-class', type=int, prompt=True, default=1, help='GradCAM target class')
@click.option("--out-dir", type=str, default=None, help="Optional output dir to save results")
@click.option('--gpus', prompt="Choose GPUs[eg: 0]", type=str)
def gradcam(**args):
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        Print("CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args["gpus"])

    configures = get_items_from_file(args["config"], format="json")
    exp_dir = Path(configures.get("experiment_path", os.path.dirname(args["config"])))

    if configures.get("n_fold", 0) > 1:
        raise NotImplementedError('Donot support cross-valid experiment')

    if os.path.isfile(args["test_files"]):
        test_fpath = args["test_files"]
        test_files = get_items_from_file(args["test_files"], format="auto")
    else:
        test_fpaths = list(exp_dir.glob('test_files*'))
        if len(test_fpaths) > 0:
            test_fpath = test_fpaths[0]
            test_files = get_items_from_file(test_fpath, format="auto")
        else:
            raise ValueError(f"Test file does not exists in {exp_dir}!")

    phase = 'test'
    configures["preload"] = 0.0
    configures["phase"] = phase
    configures["experiment_path"] = exp_dir
    configures["model_path"] = get_trained_models(exp_dir)
    configures["out_dir"] = (
        check_dir(args["out_dir"])
        if args["out_dir"]
        else check_dir(exp_dir, f'GradCam@{time.strftime("%m%d_%H%M")}')
    )

    Print(f"{len(test_files)} test files", color="g")
    test_dataloader = get_dataloader(sn(**configures), test_files, phase=phase)
    engine = get_test_engine(sn(**configures), test_dataloader)

    engine.add_event_handler(
        event_name=Events.ITERATION_COMPLETED(once=1),
        handler=GradCamHandler(
            engine.network,
            args["target_layer"],
            args["target_class"],
            engine.data_loader,
            engine.prepare_batch,
            save_dir=configures["out_dir"],
            device=torch.device("cuda")
            if args['gpus'] != "-1" else torch.device("cpu"),
            logger_name=engine.logger.name
        )
    )

    engine.run()
