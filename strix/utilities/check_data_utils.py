import logging
import warnings
from pathlib import Path
from typing import List, Optional, Union

import matplotlib
import nibabel as nib
import numpy as np
import torch
from monai_ex.data import DataLoader
from monai_ex.utils import check_dir, first
from termcolor import colored
from tqdm import tqdm

from strix.configures import config as cfg
from strix.utilities.enum import FRAMEWORKS, Phases
from strix.utilities.utils import setup_logger, get_colors, get_colormaps

matplotlib.use("Agg")
import matplotlib.cm as mpl_color_map
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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


def save_2d_image_grid(
    images: torch.Tensor,
    nrow: int,
    ncol: int,
    out_dir: Union[Path, str],
    phase: str,
    dataset_name: str,
    batch_index: int,
    fnames: List,
    axis: Optional[int] = None,
    chn_idx: Optional[int] = None,
    overlap_method: Optional[str] = "mask",
    mask: Optional[torch.Tensor] = None,
    mask_class_num: int = 2,
    alpha: float = 0.7,
    line_width: float = 0.1,
):
    if axis is not None and chn_idx is not None:
        index = torch.tensor(chn_idx).to(images.device)
        images = torch.index_select(images, dim=axis, index=index)
        if mask is not None and mask.size(axis) > 1:
            mask = torch.index_select(mask, dim=axis, index=index)

    fig = plot_segmentation_masks(
        images.detach().cpu().numpy(),
        mask.detach().cpu().numpy() if mask else None,
        nrow,
        ncol,
        alpha=alpha,
        line_width=line_width,
        method=overlap_method,
        mask_class_num=mask_class_num,
        fnames=fnames,
    )


    output_fname = f"-chn{chn_idx}" if chn_idx is not None else ""

    output_path = check_dir(
        out_dir,
        dataset_name,
        f"{phase}-batch{batch_index}{output_fname}.png",
        isFile=True,
    )

    fig.savefig(str(output_path), bbox_inches="tight", pad_inches=0)
    return output_path


def save_3d_image_grid(
    images,
    axis,
    nrow,
    ncol,
    out_dir,
    phase,
    dataset_name,
    batch_index: int,
    slice_index: int,
    fnames: List,
    multichannel: bool = False,
    overlap_method: Optional[str] = "mask",
    mask=None,
    mask_class_num=2,
    alpha: float = 0.7,
    line_width: float = 0.1, 
):
    # images = np.take(images, slice_index, axis)
    index = torch.tensor(slice_index).to(images.device)
    images = torch.index_select(images, dim=axis, index=index).squeeze(axis).detach().cpu().numpy()

    if mask is not None:
        # mask = np.take(mask, slice_index, axis)
        mask = torch.index_select(mask, dim=axis, index=index).squeeze(axis).detach().cpu().numpy()

    fig = plot_segmentation_masks(
        images,
        mask,
        nrow,
        ncol,
        alpha=alpha,
        line_width=line_width,
        method=overlap_method,
        mask_class_num=mask_class_num,
        fnames=fnames,
    )

    output_fname = f"channel{slice_index}.png" if multichannel else f"slice{slice_index}.png"
    output_path = check_dir(out_dir, dataset_name, f"{phase}-batch{batch_index}", output_fname, isFile=True)
    fig.savefig(str(output_path), bbox_inches="tight", pad_inches=0)
    return output_path


def plot_segmentation_masks(
    images: np.ndarray,
    masks: Optional[np.ndarray],
    nrow: int,
    ncol: int,
    alpha: float = 0.7,
    line_width: float = 0.1,
    method: Optional[str] = "mask",
    mask_class_num: int = 2,
    fnames: Optional[List] = None,
):
    # cm = 1/2.54
    plt.close()
    ratio = images.shape[-1] / images.shape[-2]
    fig = plt.figure(figsize=(15 * ratio * ncol, 15 * nrow))
    axes = fig.subplots(nrow, ncol)
    color_num = mask_class_num

    if color_num > 0:
        colors = get_colors(color_num)
        cmap = get_colormaps(color_num)

    for i in range(nrow):
        for j in range(ncol):
            axes[i, j].axis("off")
            if i * ncol + j < images.shape[0]:
                if fnames:
                    title = str(Path(fnames[i * ncol + j]).stem)
                    axes[i, j].set_title(title, fontsize=8)

                # draw images
                axes[i, j].imshow(images[i * ncol + j, ...].squeeze(), cmap="gray", interpolation="bicubic")

                # draw masks if exists
                if method == "mask" and masks is not None:
                    axes[i, j].imshow(
                        np.ma.masked_equal(masks[i * ncol + j, ...].squeeze(), 0),
                        cmap, #! unbound!
                        alpha=alpha,
                        norm=Normalize(vmin=1, vmax=color_num),
                    )
                elif method == "contour" and masks is not None:
                    list = [i for i in np.unique(masks[i * ncol + j, ...].squeeze()).tolist() if i]
                    if len(list) > 0:
                        axes[i, j].contour(
                            masks[i * ncol + j, ...].squeeze(),
                            levels=[x - 0.01 for x in list],
                            colors=colors[min(list) - 1 : max(list)],
                            linewidths = line_width
                        )
                    else:
                        continue

    plt.subplots_adjust(left=0.1, right=0.2, bottom=0.1, top=0.2, wspace=0.1, hspace=0.2)
    return fig


def check_dataloader(
    phase: Phases,
    dataloader: DataLoader,
    mask_key: str,
    out_dir: Union[Path, str],
    dataset_name: str,
    overlap_method: Optional[str] = None,
    alpha: float = 0.7,
    line_width: float = 0.1,
    save_raw: bool = False,
    logger: logging.Logger = logging.getLogger("data-check"),
) -> None:
    logger_name = logger.name
    first_batch = first(dataloader)
    img_key = cfg.get_key("IMAGE")
    msk_key = cfg.get_key("MASK") if mask_key is None else mask_key
    data_shape = first_batch[img_key][0].shape
    exist_mask = first_batch.get(msk_key) is not None
    channel, shape = data_shape[0], data_shape[1:]
    logger.info(f"Data Channel: {channel}, Shape: {shape}")

    if overlap_method and not exist_mask:
        logger.warn(f"{msk_key} is not found in datalist.")

    def _check_mask(masks, fnames):
        mask_class_num = len(masks.unique()) - 1
        msk = masks if mask_class_num > 0 else None
        for i in range(masks.shape[0]):
            n_class = len(msk[i, ...].unique()) - 1  #! msk is None
            if n_class == mask_class_num:
                continue
            elif n_class > 0:
                warnings.warn(
                    colored(
                        f"Other cases had {mask_class_num} kinds of labels, but {fnames[i]} got {n_class}, please check your data",
                        "yellow",
                    )
                )
            else:
                warnings.warn(
                    colored(f"Case {fnames[i]} has no label (only background), please check your data", "yellow")
                )
        return mask_class_num, msk

    if len(shape) == 2 and channel == 1:
        for i, data in enumerate(tqdm(dataloader)):
            bs = dataloader.batch_size
            fnames = data[str(img_key) + "_meta_dict"]["filename_or_obj"]
            if exist_mask and overlap_method:
                mask_class_num, msk = _check_mask(data[msk_key], fnames)
            else:
                mask_class_num = 0
                msk = None
            row = int(np.ceil(np.sqrt(bs)))
            column = row
            if (row - 1) * column >= bs:
                row -= 1
            save_2d_image_grid(
                data[img_key],
                row,
                column,
                out_dir,
                phase.value,
                dataset_name,
                i,
                fnames=fnames,
                overlap_method=overlap_method,
                mask=msk,
                mask_class_num=mask_class_num,
                alpha=alpha,
                line_width=line_width
            )
            if save_raw:
                save_raw_image(
                    data[img_key], data[f"{img_key}_meta_dict"], out_dir, phase.value, dataset_name, i, logger_name
                )

    elif len(shape) == 2 and channel > 1:
        z_axis = 1

        for i, data in enumerate(tqdm(dataloader)):
            bs = dataloader.batch_size or 1  # prevent None
            fnames = data[str(img_key) + "_meta_dict"]["filename_or_obj"]
            if exist_mask and overlap_method:
                mask_class_num, msk = _check_mask(data[msk_key], fnames)
            else:
                mask_class_num = 0
                msk = None

            row = int(np.ceil(np.sqrt(bs)))
            column = row
            if (row - 1) * column >= bs:
                row -= 1
            if save_raw:
                save_raw_image(
                    data[img_key], data[f"{img_key}_meta_dict"], out_dir, phase.value, dataset_name, i, logger_name
                )

            for ch_idx in range(channel):
                save_2d_image_grid(
                    data[img_key],
                    row,
                    column,
                    out_dir,
                    phase.value,
                    dataset_name,
                    i,
                    fnames=fnames,
                    axis=z_axis,
                    chn_idx=ch_idx,
                    overlap_method=overlap_method,
                    mask=msk,
                    mask_class_num=mask_class_num,
                    alpha=alpha,
                    line_width=line_width
                )

    elif len(shape) == 3 and channel == 1:
        z_axis = np.argmin(shape)

        for i, data in enumerate(tqdm(dataloader)):
            bs = dataloader.batch_size or 1
            fnames = data[str(img_key) + "_meta_dict"]["filename_or_obj"]
            if exist_mask and overlap_method:
                mask_class_num, msk = _check_mask(data[msk_key], fnames)
            else:
                mask_class_num = 0
                msk = None

            row = int(np.ceil(np.sqrt(bs)))
            column = row
            if (row - 1) * column >= bs:
                row -= 1
            for slice_idx in range(shape[z_axis]):
                save_3d_image_grid(
                    data[img_key],
                    z_axis + 2,
                    row,
                    column,
                    out_dir,
                    phase.value,
                    dataset_name,
                    i,
                    slice_idx,
                    fnames=fnames,
                    multichannel=False,
                    overlap_method=overlap_method,
                    mask=msk,
                    mask_class_num=mask_class_num,
                    alpha=alpha,
                    line_width=line_width
                )

            if save_raw:
                save_raw_image(
                    data[img_key], data[f"{img_key}_meta_dict"], out_dir, phase.value, dataset_name, i, logger_name
                )

    else:
        raise NotImplementedError(f"Not implement data-checking for shape of {shape}, channel of {channel}")
