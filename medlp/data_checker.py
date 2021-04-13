import click
import numpy as np
from types import SimpleNamespace as sn
from sklearn.model_selection import train_test_split
from utils_cw import get_items_from_file, Print, check_dir

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
from torchvision.utils import make_grid, save_image

from medlp.utilities.click_callbacks import data_select
from medlp.data_io import DATASET_MAPPING
from medlp.utilities.enum import FRAMEWORK_TYPES, OUTPUT_DIR
from monai_ex.utils import first
from monai_ex.data import DataLoader
from monai_ex.utils import optional_import


def save_2d_image_grid(images, nrow, out_dir, phase, dataset_name, batch_index):
    save_image(
        images,
        check_dir(out_dir, dataset_name, f'{phase}-batch{batch_index}.png', isFile=True),
        nrow=nrow,
        padding=5,
        normalize=True
    )


def save_3d_image_grid(images, axis, nrow, out_dir, phase, dataset_name, batch_index, slice_index):
    data_slice = torch.index_select(
        images, dim=axis, index=torch.tensor(slice_index)
    ).squeeze(axis)

    save_image(
        data_slice,
        check_dir(out_dir, dataset_name, f'{phase}-batch{batch_index}', f'slice{slice_index}.png', isFile=True),
        nrow=nrow,
        padding=5,
        normalize=True
    )


@click.command("check-data", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@click.option("--tensor-dim", prompt=True, type=click.Choice(["2D", "3D"]), default="2D", help="2D or 3D")
@click.option("--framework", prompt=True, type=click.Choice(FRAMEWORK_TYPES), default="segmentation", help="Choose your framework type")
@click.option("--data-list", type=str, callback=data_select, default=None, help="Data file list")
@click.option("--n-batch", prompt=True, type=int, default=10, help="Batch size")
@click.option("--split", type=float, default=0.2, help="Training/testing split ratio")
@click.option("--seed", type=int, default=101, help="random seed")
@click.option("--out-dir", type=str, prompt=True, default=OUTPUT_DIR)
@click.pass_context
def check_data(ctx, **args):
    cargs = sn(**args)
    cargs.out_dir = check_dir(cargs.out_dir, 'Data check')
    dataset_type = DATASET_MAPPING[cargs.framework][cargs.tensor_dim][cargs.data_list]
    dataset_list = DATASET_MAPPING[cargs.framework][cargs.tensor_dim][cargs.data_list + "_fpath"]
    files_list = get_items_from_file(dataset_list, format="auto")

    files_train, files_valid = train_test_split(files_list, test_size=cargs.split, random_state=cargs.seed)

    auxilary_params = {
        (ctx.args[i][2:] if str(ctx.args[i]).startswith("--") else ctx.args[i][1:]): ctx.args[i + 1]
        for i in range(0, len(ctx.args), 2)
    }

    try:
        train_ds = dataset_type(files_train, 'train', auxilary_params)
        valid_ds = dataset_type(files_valid, 'valid', auxilary_params)
    except Exception as e:
        Print(f"Creating dataset '{cargs.data_list}' failed! \nMsg: {e}", color='r')
        return
    else:
        Print(f"Creating dataset '{cargs.data_list}' successfully!", color='g')

    train_num = min(cargs.n_batch, len(train_ds))
    valid_num = min(cargs.n_batch, len(valid_ds))
    train_dataloader = DataLoader(train_ds, num_workers=5, batch_size=train_num, shuffle=True)
    valid_dataloader = DataLoader(valid_ds, num_workers=5, batch_size=valid_num, shuffle=True)
    train_data = first(train_dataloader)
    # valid_data = first(valid_dataloader)

    image_key = 'image'
    data_shape = train_data[image_key][0].shape
    data_dtype = train_data[image_key][0].dtype
    channel, shape = data_shape[0], data_shape[1:]
    if len(shape) == 2:
        for phase, dataloader in {'train': train_dataloader, 'valid': valid_dataloader}.items():
            for i, data in enumerate(dataloader):
                bs = dataloader.batch_size
                save_2d_image_grid(
                    data[image_key],
                    int(np.round(np.sqrt(bs))),
                    cargs.out_dir,
                    phase,
                    cargs.data_list,
                    i
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
