import os
import sys
import traceback
import yaml
import click
import numpy as np
import nibabel as nib
from tqdm import tqdm
from types import SimpleNamespace as sn
from sklearn.model_selection import train_test_split
from utils_cw import get_items_from_file, Print, check_dir

import torch
from torchvision.utils import save_image

from medlp.utilities.arguments import data_select
from medlp.utilities.click_callbacks import NumericChoice as Choice
from medlp.utilities.click_callbacks import get_unknown_options
from medlp.utilities.enum import FRAMEWORKS, Phases
from medlp.utilities.utils import (
    draw_segmentation_masks,
    draw_segmentation_contour,
    norm_tensor,
    get_colors,
)
from medlp.data_io import DATASET_MAPPING
from medlp.configures import config as cfg
from monai_ex.utils import first
from monai_ex.data import DataLoader


def save_raw_image(data, meta_dict, out_dir, phase, dataset_name, batch_index):
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
    out_dir,
    phase,
    dataset_name,
    batch_index,
    axis=None,
    chn_idx=None,
    overlap_method=draw_segmentation_masks,
    mask=None,
):
    if axis is not None and chn_idx is not None:
        images = torch.index_select(images, dim=axis, index=torch.tensor(chn_idx))

        if mask is not None and mask.size(axis) > 1:
            mask = torch.index_select(mask, dim=axis, index=torch.tensor(chn_idx))

    if mask is not None:
        data_slice = norm_tensor(images, None).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        mask_slice = mask.to(torch.bool)
        mask_num_classes = len(mask_slice.unique())

        images = torch.cat(
            [
                overlap_method(img, msk, 0.6, colors=get_colors(max(msk.size()[0], mask_num_classes))).unsqueeze(0)
                for img, msk in zip(data_slice, mask_slice)
            ]
        ).float()

    output_fname = f"-chn{chn_idx}" if chn_idx is not None else ""

    output_path = check_dir(
        out_dir,
        dataset_name,
        f"{phase}-batch{batch_index}{output_fname}.png",
        isFile=True,
    )

    save_image(images, output_path, nrow=nrow, padding=5, normalize=True, scale_each=True)

    return output_path


def save_3d_image_grid(
    images,
    axis,
    nrow,
    out_dir,
    phase,
    dataset_name,
    batch_index,
    slice_index,
    multichannel=False,
    overlap_method=draw_segmentation_masks,
    mask=None,
):
    data_slice = torch.index_select(images, dim=axis, index=torch.tensor(slice_index)).squeeze(axis)

    if mask is not None:
        mask_slice = torch.index_select(mask, dim=axis, index=torch.tensor(slice_index)).squeeze(axis)

        data_slice = norm_tensor(data_slice, None).mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
        mask_slice = mask_slice.to(torch.bool)
        mask_num_classes = len(mask_slice.unique())

        data_slice = torch.cat(
            [
                overlap_method(img, msk, 0.6, colors=get_colors(max(msk.size()[0], mask_num_classes))).unsqueeze(0)
                for img, msk in zip(data_slice, mask_slice)
            ]
        ).float()

    if multichannel:
        output_fname = f"channel{slice_index}.png"
    else:
        output_fname = f"slice{slice_index}.png"

    output_path = check_dir(out_dir, dataset_name, f"{phase}-batch{batch_index}", output_fname, isFile=True)
    save_image(data_slice, output_path, nrow=nrow, padding=5, normalize=True, scale_each=True)

    return output_path


@click.command("check-data", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.option("--tensor-dim", prompt=True, type=Choice(["2D", "3D"]), default="2D", help="2D or 3D")
@click.option(
    "--framework", prompt=True, type=Choice(FRAMEWORKS), default="segmentation", help="Choose your framework type"
)
@click.option("--data-list", type=str, callback=data_select, default=None, help="Data file list")
@click.option("--n-batch", prompt=True, type=int, default=9, help="Batch size")
@click.option("--split", type=float, default=0.2, help="Training/testing split ratio")
@click.option("--save-raw", is_flag=True, help="Save processed raw image to local")
@click.option("--mask-overlap", is_flag=True, help="Overlapping mask if exists")
@click.option("--contour-overlap", is_flag=True, help="Overlapping mask's contour")
@click.option("--mask-key", type=str, default="mask", help="Specify mask key, default is 'mask'")
@click.option("--seed", type=int, default=101, help="random seed")
@click.option("--out-dir", type=str, prompt=True, default=cfg.get_medlp_cfg("OUTPUT_DIR"))
@click.pass_context
def check_data(ctx, **args):
    cargs = sn(**args)
    auxilary_params = get_unknown_options(ctx, verbose=True)
    cargs.out_dir = check_dir(cargs.out_dir, "data checking")
    auxilary_params.update({"experiment_path": str(cargs.out_dir)})

    data_attr = DATASET_MAPPING[cargs.framework][cargs.tensor_dim][cargs.data_list]
    dataset_fn, dataset_list = data_attr["FN"], data_attr["PATH"]
    files_list = get_items_from_file(dataset_list, format="auto")

    files_train, files_valid = train_test_split(files_list, test_size=cargs.split, random_state=cargs.seed)

    try:
        train_ds = dataset_fn(files_train, Phases.TRAIN, auxilary_params)
        valid_ds = dataset_fn(files_valid, Phases.VALID, auxilary_params)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        Print(f"Creating dataset '{cargs.data_list}' failed! \nMsg: {repr(e)}", color="r")
        Print("Exception trace:", color="r")
        print("\n".join(traceback.format_tb(exc_tb)))
        return
    else:
        Print(f"Creating dataset '{cargs.data_list}' successfully!", color="g")

    train_num = min(cargs.n_batch, len(train_ds))
    valid_num = min(cargs.n_batch, len(valid_ds))
    train_dataloader = DataLoader(train_ds, num_workers=1, batch_size=train_num, shuffle=True)
    valid_dataloader = DataLoader(valid_ds, num_workers=1, batch_size=valid_num, shuffle=False)
    train_data = first(train_dataloader)
    # valid_data = first(valid_dataloader)

    img_key = cfg.get_key("IMAGE")
    msk_key = cfg.get_key("MASK") if cargs.mask_key is None else cargs.mask_key
    data_shape = train_data[img_key][0].shape
    exist_mask = train_data.get(msk_key) is not None
    channel, shape = data_shape[0], data_shape[1:]
    print("Channel:", channel, "Shape:", shape)

    if cargs.mask_overlap and cargs.contour_overlap:
        raise ValueError("mask_overlap/contour_overlap can only choose one!")
    overlap = cargs.mask_overlap or cargs.contour_overlap

    if overlap and not exist_mask:
        Print(f"{msk_key} is not found in datalist.", color="y")

    if cargs.mask_overlap:
        overlap_m = draw_segmentation_masks
    elif cargs.contour_overlap:
        overlap_m = draw_segmentation_contour
    else:
        overlap_m = draw_segmentation_masks

    if len(shape) == 2 and channel == 1:
        for phase, dataloader in {
            Phases.TRAIN.value: train_dataloader,
            Phases.VALID.value: valid_dataloader,
        }.items():
            for i, data in enumerate(tqdm(dataloader)):
                bs = dataloader.batch_size
                msk = data[msk_key] if exist_mask and overlap else None
                output_fpath = save_2d_image_grid(
                    data[img_key],
                    int(np.ceil(np.sqrt(bs))),
                    cargs.out_dir,
                    phase,
                    cargs.data_list,
                    i,
                    overlap_method=overlap_m,
                    mask=msk,
                )

                if cargs.save_raw:
                    if isinstance(data[img_key], torch.Tensor):
                        out_data = data[img_key].cpu().numpy()
                    else:
                        out_data = data[img_key]

                    save_raw_image(
                        out_data,
                        data[f"{img_key}_meta_dict"],
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
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
                msk = data[msk_key] if exist_mask and overlap else None
                if cargs.save_raw:
                    if isinstance(data[img_key], torch.Tensor):
                        out_data = data[img_key].cpu().numpy()
                    else:
                        out_data = data[img_key]

                    save_raw_image(
                        out_data,
                        data[f"{img_key}_meta_dict"],
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
                    )

                for ch_idx in range(channel):
                    output_fpath = save_2d_image_grid(
                        data[img_key],
                        int(np.ceil(np.sqrt(bs))),
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
                        z_axis,
                        ch_idx,
                        overlap_method=overlap_m,
                        mask=msk,
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
                msk = data[msk_key] if exist_mask and overlap else None
                for slice_idx in range(shape[z_axis]):
                    output_fpath = save_3d_image_grid(
                        data[img_key],
                        z_axis + 2,
                        int(np.ceil(np.sqrt(bs))),
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
                        slice_idx,
                        multichannel=False,
                        overlap_method=overlap_m,
                        mask=msk,
                    )

                save_fnames(data, img_key + "_meta_dict", output_fpath)

    else:
        raise NotImplementedError(f"Not implement data-checking for shape of {shape}, channel of {channel}")
