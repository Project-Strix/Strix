import os
import click
from pathlib import Path
from functools import partial
from types import SimpleNamespace as sn
from sklearn.model_selection import train_test_split

import torch

from strix.utilities.arguments import data_select
from strix.utilities.click import OptionEx, CommandEx
from strix.utilities.click import NumericChoice as Choice
from strix.utilities.click_callbacks import get_unknown_options, dump_params, parse_project
from strix.utilities.enum import FRAMEWORKS, Phases, SerialFileFormat
from strix.utilities.utils import (
    setup_logger,
    get_items,
    trycatch,
    generate_synthetic_datalist,
)
from strix.utilities.check_data_utils import check_dataloader
from strix.utilities.registry import DatasetRegistry
from strix.configures import config as cfg
from monai_ex.utils import check_dir
from monai_ex.data import DataLoader


option = partial(click.option, cls=OptionEx)
command = partial(click.command, cls=CommandEx)
check_cmd_history = os.path.join(cfg.get_strix_cfg("cache_dir"), ".strix_check_cmd_history")


@command(
    "check-data",
    context_settings={
        "allow_extra_args": True,
        "ignore_unknown_options": True,
        "prompt_in_default_map": True,
        "default_map": get_items(check_cmd_history, format=SerialFileFormat.YAML, allow_filenotfound=True),
    },
)
@option("--tensor-dim", prompt=True, type=Choice(["2D", "3D"]), default="2D", help="2D or 3D")
@option("--framework", prompt=True, type=Choice(FRAMEWORKS), default="segmentation", help="Choose your framework type")
@option("--alpha", type=float, default=0.7, help="The opacity of mask")
@option("--project", type=click.Path(), callback=parse_project, default=Path.cwd(), help="Project folder path")
@option("--data-list", type=str, callback=data_select, default=None, help="Data file list")  # todo: Rename
@option("--n-batch", prompt=True, type=int, default=9, help="Batch size")
@option("--split", type=float, default=0.2, help="Training/testing split ratio")
@option("--save-raw", is_flag=True, help="Save processed raw image to local")
@option("--mask-overlap", is_flag=True, help="Overlapping mask if exists")
@option("--contour-overlap", is_flag=True, help="Overlapping mask's contour")
@option("--mask-key", type=str, default="mask", help="Specify mask key, default is 'mask'")
@option("--seed", type=int, default=101, help="random seed")
@option("--out-dir", type=str, prompt=True, default=cfg.get_strix_cfg("OUTPUT_DIR"))
@option(
    "--dump-params",
    hidden=True,
    is_flag=True,
    default=False,
    callback=partial(dump_params, output_path=check_cmd_history),
)
@click.pass_context
def check_data(ctx, **args):
    cargs = sn(**args)
    auxilary_params = get_unknown_options(ctx, verbose=True)
    cargs.out_dir = check_dir(cargs.out_dir, "data checking")
    auxilary_params.update({"experiment_path": str(cargs.out_dir)})
    auxilary_params.update(**args)

    logger_name = f"check-{cargs.data_list}"
    logger = setup_logger(logger_name)

    @trycatch(show_details=False)
    def get_train_valid_datasets():
        datasets = DatasetRegistry()
        strix_dataset = datasets.get(cargs.tensor_dim, cargs.framework, cargs.data_list)
        if strix_dataset is None:
            raise ValueError(f"{cargs.data_list} not found!")
        dataset_fn, dataset_list = strix_dataset.get("FN"), strix_dataset.get("PATH")

        if dataset_list is None:  # synthetic data
            file_list = generate_synthetic_datalist(100, logger)
        else:
            file_list = get_items(dataset_list)
        files_train, files_valid = train_test_split(file_list, test_size=cargs.split, random_state=cargs.seed)

        train_ds = dataset_fn(files_train, Phases.TRAIN, auxilary_params)
        valid_ds = dataset_fn(files_valid, Phases.VALID, auxilary_params)
        return train_ds, valid_ds

    train_dataset, valid_dataset = get_train_valid_datasets()
    logger.info(f"Creating dataset '{cargs.data_list}' successfully!")

    if isinstance(train_dataset, torch.utils.data.DataLoader):
        train_dataloader = train_dataset
        valid_dataloader = valid_dataset
    else:
        train_num = min(cargs.n_batch, len(train_dataset))
        valid_num = min(cargs.n_batch, len(valid_dataset))
        train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=train_num, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, num_workers=1, batch_size=valid_num, shuffle=False)

    if cargs.mask_overlap and cargs.contour_overlap:
        raise ValueError("mask_overlap/contour_overlap can only choose one!")

    overlap_m = "mask" if cargs.mask_overlap else "contour" if cargs.contour_overlap else None

    check_dataloader(
        phase=Phases.TRAIN,
        dataloader=train_dataloader,
        mask_key=cargs.mask_key,
        out_dir=cargs.out_dir,
        dataset_name=cargs.data_list,
        overlap_method=overlap_m,
        alpha=cargs.alpha,
        save_raw=cargs.save_raw,
        logger=logger,
    )

    check_dataloader(
        phase=Phases.VALID,
        dataloader=valid_dataloader,
        mask_key=cargs.mask_key,
        out_dir=cargs.out_dir,
        dataset_name=cargs.data_list,
        overlap_method=overlap_m,
        alpha=cargs.alpha,
        save_raw=cargs.save_raw,
        logger=logger,
    )
