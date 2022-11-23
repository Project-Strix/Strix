import os
import yaml
import click
import numpy as np
import nibabel as nib
from tqdm import tqdm
from pathlib import Path
from functools import partial
from types import SimpleNamespace as sn
from sklearn.model_selection import train_test_split
import warnings
from termcolor import colored

import torch

from strix.utilities.arguments import data_select
from strix.utilities.click import OptionEx, CommandEx
from strix.utilities.click import NumericChoice as Choice
from strix.utilities.click_callbacks import get_unknown_options, dump_params, parse_project
from strix.utilities.enum import FRAMEWORKS, Phases, SerialFileFormat
from strix.utilities.utils import (
    plot_segmentation_masks,
    setup_logger,
    get_items,
    trycatch,
    generate_synthetic_datalist,
)
from strix.utilities.registry import DatasetRegistry
from strix.configures import config as cfg
from monai_ex.utils import first, check_dir
from monai_ex.data import DataLoader


def save_raw_image(data, meta_dict, out_dir, phase, dataset_name, batch_index, logger_name=None):
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    if logger_name:
        logger = setup_logger(logger_name)
        logger.info(f"Saving {phase} image to {out_dir}...")

    for i, patch in enumerate(data):
        out_fname = check_dir(
            out_dir,
            dataset_name,
            "raw images",
            f"{phase}-batch{batch_index}-{i}.nii.gz",
            isFile=True,
        )
        nib.save(nib.Nifti1Image(patch.squeeze(), meta_dict["affine"][i]), out_fname)


def save_fnames(data, img_meta_key, image_fpath):
    fnames = {idx + 1: fname for idx, fname in enumerate(data[img_meta_key]["filename_or_obj"])}
    image_fpath = str(image_fpath).split("-chn")[0]
    output_path = os.path.splitext(image_fpath)[0] + "-fnames.yml"
    with open(output_path, "w") as f:
        yaml.dump(fnames, f)


def save_2d_image_grid(
    images,
    nrow,
    ncol,
    out_dir,
    phase,
    dataset_name,
    batch_index,
    axis=None,
    chn_idx=None,
    overlap_method='mask',
    mask=None,
    mask_class_num = 2,
    fnames: list = None,
    alpha: float = None
):
    if axis is not None and chn_idx is not None:
        images = np.take(images, chn_idx, axis)
        if mask is not None and mask.size(axis) > 1:
            mask = np.take(mask, chn_idx, axis)
    if mask is not None:
        data_slice = images.detach().numpy()
        mask_slice = mask.detach().numpy()
        fig = plot_segmentation_masks(data_slice, mask_slice, nrow, ncol, 
                alpha = alpha, method = overlap_method, mask_class_num = mask_class_num, fnames = fnames)

    output_fname = f"-chn{chn_idx}" if chn_idx is not None else ""

    output_path = check_dir(
        out_dir,
        dataset_name,
        f"{phase}-batch{batch_index}{output_fname}.png",
        isFile=True,
    )

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    return output_path


def save_3d_image_grid(
    images,
    axis,
    nrow,
    ncol,
    out_dir,
    phase,
    dataset_name,
    batch_index,
    slice_index,
    multichannel=False,
    overlap_method='mask',
    mask=None,
    mask_class_num = 2,
    fnames: list = None,
    alpha: float = None
):
    images = np.take(images, slice_index, axis)
    if mask is not None:
        mask = np.take(mask, slice_index, axis)
    if mask is not None:
        data_slice = images.detach().numpy()
        mask_slice = mask.detach().numpy()
        fig = plot_segmentation_masks(data_slice, mask_slice, nrow, ncol, 
            alpha = alpha, method = overlap_method, mask_class_num=mask_class_num, fnames = fnames)

    if multichannel:
        output_fname = f"channel{slice_index}.png"
    else:
        output_fname = f"slice{slice_index}.png"

    output_path = check_dir(out_dir, dataset_name, f"{phase}-batch{batch_index}", output_fname, isFile=True)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0)
    return output_path


option = partial(click.option, cls=OptionEx)
command = partial(click.command, cls=CommandEx)
check_cmd_history = os.path.join(cfg.get_strix_cfg("cache_dir"), '.strix_check_cmd_history')


@command(
    "check-data", 
    context_settings={
        "allow_extra_args": True, "ignore_unknown_options": True, "prompt_in_default_map": True,
        "default_map": get_items(check_cmd_history, format=SerialFileFormat.YAML, allow_filenotfound=True)
    })
@option("--tensor-dim", prompt=True, type=Choice(["2D", "3D"]), default="2D", help="2D or 3D")
@option(
    "--framework", prompt=True, type=Choice(FRAMEWORKS), default="segmentation", help="Choose your framework type"
)
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
@option("--dump-params", hidden=True, is_flag=True, default=False, callback=partial(dump_params, output_path=check_cmd_history))
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

    train_num = min(cargs.n_batch, len(train_dataset))
    valid_num = min(cargs.n_batch, len(valid_dataset))
    train_dataloader = DataLoader(train_dataset, num_workers=1, batch_size=train_num, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, num_workers=1, batch_size=valid_num, shuffle=False)
    train_data = first(train_dataloader)

    img_key = cfg.get_key("IMAGE")
    msk_key = cfg.get_key("MASK") if cargs.mask_key is None else cargs.mask_key
    data_shape = train_data[img_key][0].shape
    exist_mask = train_data.get(msk_key) is not None
    channel, shape = data_shape[0], data_shape[1:]
    logger.info(f"Data Channel: {channel}, Shape: {shape}")

    if cargs.mask_overlap and cargs.contour_overlap:
        raise ValueError("mask_overlap/contour_overlap can only choose one!")
    overlap = cargs.mask_overlap or cargs.contour_overlap

    if overlap and not exist_mask:
        logger.warn(f"{msk_key} is not found in datalist.")

    if cargs.mask_overlap:
        overlap_m = 'mask'
    elif cargs.contour_overlap:
        overlap_m = 'contour'

    def _check_mask(masks, fnames):
        mask_class_num = len(masks.unique()) - 1
        msk = masks if mask_class_num > 0 else None
        for i in range(masks.shape[0]):
            n_class = len(msk[i,...].unique()) - 1
            if  n_class == mask_class_num:
                continue
            elif n_class > 0:
                warnings.warn(colored(f"Other cases had {mask_class_num} kinds of labels, but {fnames[i]} got {n_class}, please check your data", "yellow"))
            else:
                warnings.warn(colored(f"Case {fnames[i]} has no label (only background), please check your data", "yellow"))
        return mask_class_num, msk

    if len(shape) == 2 and channel == 1:
        for phase, dataloader in {
            Phases.TRAIN.value: train_dataloader,
            Phases.VALID.value: valid_dataloader,
        }.items():
            for i, data in enumerate(tqdm(dataloader)):
                bs = dataloader.batch_size
                if exist_mask and overlap: 
                    fnames = data[str(img_key) + '_meta_dict']["filename_or_obj"]
                    mask_class_num, msk = _check_mask(data[msk_key],fnames)
                else:
                    mask_class_num = 0
                    msk=None
                row = int(np.ceil(np.sqrt(bs)))
                column = row
                if (row -1) * column >= bs:
                    row -= 1
                output_fpath = save_2d_image_grid(
                    data[img_key],
                    row,
                    column,
                    cargs.out_dir,
                    phase,
                    cargs.data_list,
                    i,
                    overlap_method=overlap_m,
                    mask=msk,
                    mask_class_num = mask_class_num,
                    fnames = fnames,
                    alpha = cargs.alpha
                )
                if cargs.save_raw:                  
                    save_raw_image(
                        data[img_key],
                        data[f"{img_key}_meta_dict"],
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
                        logger_name
                    )

                save_fnames(data, img_key + "_meta_dict", output_fpath)

    elif len(shape) == 2 and channel > 1:
        z_axis = 1
        for phase, dataloader in {
            Phases.TRAIN.value: train_dataloader,
            Phases.VALID.value: valid_dataloader,
        }.items():
            for i, data in enumerate(tqdm(dataloader)):
                bs = dataloader.batch_size
                if exist_mask and overlap:
                    fnames = data[str(img_key) + '_meta_dict']["filename_or_obj"]
                    mask_class_num, msk = _check_mask(data[msk_key],fnames)
                else:
                    mask_class_num = 0
                    msk=None

                row = int(np.ceil(np.sqrt(bs)))
                column = row
                if (row -1) * column >= bs:
                    row -= 1
                if cargs.save_raw:
                    save_raw_image(
                        data[img_key],
                        data[f"{img_key}_meta_dict"],
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
                        logger_name
                    )

                for ch_idx in range(channel):
                    output_fpath = save_2d_image_grid(
                        data[img_key],
                        row,
                        column,
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
                        axis = z_axis,
                        chn_idx =ch_idx,
                        overlap_method=overlap_m,
                        mask=msk,
                        mask_class_num=mask_class_num,
                        fnames = fnames,
                        alpha = cargs.alpha
                    )

                save_fnames(data, img_key + "_meta_dict", output_fpath)

    elif len(shape) == 3 and channel == 1:
        z_axis = np.argmin(shape)
        for phase, dataloader in {
            Phases.TRAIN.value: train_dataloader,
            Phases.VALID.value: valid_dataloader,
        }.items():
            for i, data in enumerate(tqdm(dataloader)):
                bs = dataloader.batch_size
                if exist_mask and overlap:
                    fnames = data[str(img_key) + '_meta_dict']["filename_or_obj"]
                    mask_class_num, msk = _check_mask(data[msk_key],fnames)
                else:
                    mask_class_num = 0
                    msk=None

                row = int(np.ceil(np.sqrt(bs)))
                column = row
                if (row -1) * column >= bs:
                    row -= 1
                for slice_idx in range(shape[z_axis]):
                    output_fpath = save_3d_image_grid(
                        data[img_key],
                        z_axis + 2,
                        row,
                        column,
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
                        slice_idx,
                        multichannel=False,
                        overlap_method=overlap_m,
                        mask=msk,
                        mask_class_num=mask_class_num,
                        fnames = fnames,
                        alpha = cargs.alpha
                    )
                
                if cargs.save_raw:
                    save_raw_image(
                        data[img_key],
                        data[f"{img_key}_meta_dict"],
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
                        logger_name
                    )

                save_fnames(data, img_key + "_meta_dict", output_fpath)

    else:
        raise NotImplementedError(f"Not implement data-checking for shape of {shape}, channel of {channel}")
