import os
import sys
import yaml
import click
import numpy as np
import nibabel as nib
from types import SimpleNamespace as sn
from sklearn.model_selection import train_test_split
from utils_cw import get_items_from_file, Print, check_dir

import torch
from torchvision.utils import make_grid, save_image

from medlp.utilities.click_callbacks import data_select
from medlp.utilities.click_ex import NumericChoice as Choice
from medlp.utilities.click_ex import get_unknown_options
from medlp.data_io import DATASET_MAPPING
from medlp.utilities.enum import FRAMEWORK_TYPES
from medlp.configures import config as cfg
from monai_ex.utils import first
from monai_ex.data import DataLoader


def save_raw_image(data, meta_dict, out_dir, phase, dataset_name, batch_index):
    for i, patch in enumerate(data):
        out_fname = check_dir(out_dir, dataset_name, 'raw images', f'{phase}-batch{batch_index}-{i}.nii.gz', isFile=True)
        nib.save( nib.Nifti1Image( patch.squeeze(), meta_dict['affine'][i] ), out_fname )


def save_fnames(fnames, out_dir, phase, dataset_name, batch_index):
    out_fname = check_dir(out_dir, dataset_name, f'{phase}-batch{batch_index}-fnames.yml', isFile=True)
    with out_fname.open('w') as f:
        yaml.dump(fnames, f)


def save_2d_image_grid(images, nrow, out_dir, phase, dataset_name, batch_index, axis=None, chn_idx=None):
    data_slice = None
    if axis is not None and chn_idx is not None:
        data_slice = torch.index_select(
            images, dim=axis, index=torch.tensor(chn_idx)
        )  #.squeeze(axis)

    output_fname = f'_chn{chn_idx}' if chn_idx is not None else ''

    save_image(
        images if data_slice is None else data_slice,
        check_dir(out_dir, dataset_name, f'{phase}-batch{batch_index}{output_fname}.png', isFile=True),
        nrow=nrow,
        padding=5,
        normalize=True
    )


def save_3d_image_grid(images, axis, nrow, out_dir, phase, dataset_name, batch_index, slice_index, multichannel=False):
    data_slice = torch.index_select(
        images, dim=axis, index=torch.tensor(slice_index)
    ).squeeze(axis)

    if multichannel:
        output_fname = f'channel{slice_index}.png'
    else:
        output_fname = f'slice{slice_index}.png'

    save_image(
        data_slice,
        check_dir(out_dir, dataset_name, f'{phase}-batch{batch_index}', output_fname, isFile=True),
        nrow=nrow,
        padding=5,
        normalize=True
    )


@click.command("check-data", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.option("--tensor-dim", prompt=True, type=Choice(["2D", "3D"]), default="2D", help="2D or 3D")
@click.option("--framework", prompt=True, type=Choice(FRAMEWORK_TYPES), default="segmentation", help="Choose your framework type")
@click.option("--data-list", type=str, callback=data_select, default=None, help="Data file list")
@click.option("--n-batch", prompt=True, type=int, default=10, help="Batch size")
@click.option("--split", type=float, default=0.2, help="Training/testing split ratio")
@click.option("--save-raw", is_flag=True, help="Save processed raw image to local")
@click.option("--seed", type=int, default=101, help="random seed")
@click.option("--out-dir", type=str, prompt=True, default=cfg.get_medlp_cfg("OUTPUT_DIR"))
@click.pass_context
def check_data(ctx, **args):
    cargs = sn(**args)
    auxilary_params = get_unknown_options(ctx, verbose=True)
    cargs.out_dir = check_dir(cargs.out_dir, 'data checking')

    dataset_type = DATASET_MAPPING[cargs.framework][cargs.tensor_dim][cargs.data_list]
    dataset_list = DATASET_MAPPING[cargs.framework][cargs.tensor_dim][cargs.data_list + "_fpath"]
    files_list = get_items_from_file(dataset_list, format="auto")

    files_train, files_valid = train_test_split(files_list, test_size=cargs.split, random_state=cargs.seed)

    try:
        train_ds = dataset_type(files_train, 'train', auxilary_params)
        valid_ds = dataset_type(files_valid, 'valid', auxilary_params)
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        Print(f"Creating dataset '{cargs.data_list}' failed! \nMsg: {e}", color='r')
        Print(f"Error location: {fname}, line:{exc_tb.tb_lineno}", color='r')
        return
    else:
        Print(f"Creating dataset '{cargs.data_list}' successfully!", color='g')

    train_num = min(cargs.n_batch, len(train_ds))
    valid_num = min(cargs.n_batch, len(valid_ds))
    train_dataloader = DataLoader(train_ds, num_workers=1, batch_size=train_num, shuffle=True)
    valid_dataloader = DataLoader(valid_ds, num_workers=1, batch_size=valid_num, shuffle=False)
    train_data = first(train_dataloader)
    # valid_data = first(valid_dataloader)

    image_key = cfg.get_key("IMAGE")
    data_shape = train_data[image_key][0].shape
    data_dtype = train_data[image_key][0].dtype
    channel, shape = data_shape[0], data_shape[1:]
    print("Channel:", channel, "Shape:", shape)
    if len(shape) == 2 and channel == 1:
        for phase, dataloader in {'train': train_dataloader, 'valid': valid_dataloader}.items():
            for i, data in enumerate(dataloader):
                bs = dataloader.batch_size
                save_2d_image_grid(
                    data[image_key],
                    int(np.round(np.sqrt(bs))),
                    cargs.out_dir,
                    phase,
                    cargs.data_list,
                    i,
                )
                fnames = {idx+1: fname for idx, fname in enumerate(data[image_key+'_meta_dict']['filename_or_obj'])}
                save_fnames(fnames, cargs.out_dir, phase, cargs.data_list, i)

    elif len(shape) == 2 and channel > 1:
        z_axis = 1
        for phase, dataloader in {'train': train_dataloader, 'valid': valid_dataloader}.items():
            for i, data in enumerate(dataloader):
                bs = dataloader.batch_size
                if cargs.save_raw:
                    if isinstance(data[image_key], torch.Tensor):
                        out_data = data[image_key].cpu().numpy()
                    else:
                        out_data = data[image_key]

                    save_raw_image(
                        out_data,
                        data[f'{image_key}_meta_dict'],
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i
                    )

                for ch_idx in range(channel):
                    save_2d_image_grid(
                        data[image_key],
                        int(np.round(np.sqrt(bs))),
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
                        z_axis,
                        ch_idx
                    )

    elif len(shape) == 3 and channel == 1:
        z_axis = np.argmin(shape)
        for phase, dataloader in {'train': train_dataloader, 'valid': valid_dataloader}.items():
            for i, data in enumerate(dataloader):
                bs = dataloader.batch_size
                for slice_idx in range(shape[z_axis]):
                    save_3d_image_grid(
                        data[image_key],
                        z_axis+2,
                        int(np.round(np.sqrt(bs))),
                        cargs.out_dir,
                        phase,
                        cargs.data_list,
                        i,
                        slice_idx
                    )

    else:
        raise NotImplementedError
