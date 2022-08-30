import os
import time
import json
import yaml
import socket
import struct
import logging
import warnings
from pathlib import Path
from functools import partial
from termcolor import colored
from typing import List, Optional, Tuple, Union, TextIO, Sequence, Dict

import matplotlib
import pylab
import torch
from PIL import Image, ImageColor, ImageDraw

matplotlib.use("Agg")
import matplotlib.cm as mpl_color_map
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tensorboard.compat.proto.event_pb2 as event_pb2
from matplotlib.ticker import ScalarFormatter
from strix.utilities.enum import LR_SCHEDULES, Phases, DatalistKeywords
from monai.networks import one_hot
from monai_ex.utils import ensure_list, GenericException
from utils_cw import catch_exception, get_items_from_file

trycatch = partial(catch_exception, handled_exception_type=GenericException, path_keywords="strix")


@trycatch()
def get_items(filelist, format="auto", sep="\n", allow_filenotfound: bool = False):
    """Wrapper of utils_cw's `get_items_from_file` function with `trycatch` decorator."""
    try:
        return get_items_from_file(filelist, format, sep)
    except json.JSONDecodeError as e:
        raise GenericException("Content of your json file cannot be parsed. Please recheck it!")
    except yaml.YAMLError as e:
        if hasattr(e, "problem_mark"):
            mark = e.problem_mark
        raise GenericException(
            f"Content of your yaml file cannot be parsed. Please recheck it!\n"
            f"Error parsing at line {mark.line}, column {mark.column+1}."
        )
    except TypeError as e:
        raise GenericException("Your data path not exists! Please recheck!") from e
    except FileNotFoundError as e:
        if allow_filenotfound:
            return None
        else:
            raise GenericException(f"File not found! {filelist}")


def generate_synthetic_datalist(data_num: int = 100, logger=None):
    if logger:
        logger.info(" ==== Using synthetic data ====")
    train_datalist = [
        {"image": f"synthetic_image{i}.nii.gz", "label": f"synthetic_label{i}.nii.gz"} for i in range(data_num)
    ]
    return train_datalist


@trycatch()
def parse_datalist(filelist, format="auto", include_unlabel: bool = False):
    """Wrapper of `get_items` function for parsing datalist

    Args:
        filelist (list): input filelist containing data path.
        format (str, optional): datalist file format. Defaults to "auto".
        include_unlabel (bool, optional): whether return unlabeled data if exists. Defaults to False.
    """
    datalist = get_items(filelist, format=format)
    if isinstance(datalist, Sequence):
        return datalist  # normal datalist
    elif isinstance(datalist, Dict):
        label: str = DatalistKeywords.LABEL.value
        unlabel: str = DatalistKeywords.UNLABEL.value
        if label not in datalist:
            raise GenericException(f"Your datalist does not contain '{label}' key.")
        if include_unlabel and unlabel not in datalist:
            raise GenericException(
                f"Your datalist does not contain '{unlabel}' key, but you need it for your task!"
            )
        if include_unlabel:
            return datalist[label], datalist[unlabel]
        return datalist[label]
    else:
        return GenericException(
            f"Current datalist should be list or dict (labeled&unlabeled), but got {type(datalist)}"
        )


def get_attr_(obj, name, default = None):
    return getattr(obj, name) if hasattr(obj, name) else default


def get_colors(num: int = None):
    if num:
        return list(mcolors.TABLEAU_COLORS.values())[:num]
    else:
        return list(mcolors.TABLEAU_COLORS.values())


def bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    return rmin, rmax, cmin, cmax, zmin, zmax


def bbox_2D(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap * 255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert("RGBA"))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def create_rgb_summary(label):
    num_colors = label.shape[1]

    cm = pylab.get_cmap("gist_rainbow")

    new_label = np.zeros((label.shape[0], label.shape[1], label.shape[2], 3), dtype=np.float32)

    for i in range(num_colors):
        color = cm(1.0 * i / num_colors)  # color will now be an RGBA tuple
        new_label[:, :, :, 0] += label[:, :, :, i] * color[0]
        new_label[:, :, :, 1] += label[:, :, :, i] * color[1]
        new_label[:, :, :, 2] += label[:, :, :, i] * color[2]

    return new_label


def add_3D_overlay_to_summary(
    data: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    writer,
    index: int = 0,
    tag: str = "output",
    centers=None,
):
    data_ = data[index].detach().cpu().numpy() if torch.is_tensor(data) else data[index]
    mask_ = mask[index].detach().cpu().numpy() if torch.is_tensor(mask) else mask[index]
    # binary_volume = np.squeeze(binary_volume)
    # volume_overlay = np.squeeze(volume_overlay)

    if mask_.shape[1] > 1:
        # there are channels
        mask_ = create_rgb_summary(mask_)
        data_ = data_[..., np.newaxis]

    else:
        data_, mask_ = data_[..., np.newaxis], mask_[..., np.newaxis]

    if centers is None:
        center_x = np.argmax(
            np.sum(
                np.sum(np.sum(mask_, axis=3, keepdims=True), axis=2, keepdims=True),
                axis=1,
                keepdims=True,
            ),
            axis=0,
        )
        center_y = np.argmax(
            np.sum(
                np.sum(np.sum(mask_, axis=3, keepdims=True), axis=2, keepdims=True),
                axis=0,
                keepdims=True,
            ),
            axis=1,
        )
        center_z = np.argmax(
            np.sum(
                np.sum(np.sum(mask_, axis=3, keepdims=True), axis=1, keepdims=True),
                axis=0,
                keepdims=True,
            ),
            axis=2,
        )
    else:
        center_x, center_y, center_z = centers

    segmentation_overlay_x = np.squeeze(data_[center_x, :, :, :] + mask_[center_x, :, :, :])
    segmentation_overlay_y = np.squeeze(data_[:, center_y, :, :] + mask_[:, center_y, :, :])
    segmentation_overlay_z = np.squeeze(data_[:, :, center_z, :] + mask_[:, :, center_z, :])

    if len(segmentation_overlay_x.shape) != 3:
        segmentation_overlay_x, segmentation_overlay_y, segmentation_overlay_z = (
            segmentation_overlay_x[..., np.newaxis],
            segmentation_overlay_y[..., np.newaxis],
            segmentation_overlay_z[..., np.newaxis],
        )

    writer.add_image(tag + "_x", segmentation_overlay_x)
    writer.add_image(tag + "_y", segmentation_overlay_y)
    writer.add_image(tag + "_z", segmentation_overlay_z)


def add_3D_image_to_summary(manager, image, name, centers=None):
    image = np.squeeze(image)

    if len(image.shape) > 3:
        # there are channels
        print("add_3D_image_to_summary: there are channels")
        image = create_rgb_summary(image)
    else:
        image = image[..., np.newaxis]

    if centers is None:
        center_x = np.argmax(
            np.sum(
                np.sum(np.sum(image, axis=3, keepdims=True), axis=2, keepdims=True),
                axis=1,
                keepdims=True,
            ),
            axis=0,
        )
        center_y = np.argmax(
            np.sum(
                np.sum(np.sum(image, axis=3, keepdims=True), axis=2, keepdims=True),
                axis=0,
                keepdims=True,
            ),
            axis=1,
        )
        center_z = np.argmax(
            np.sum(
                np.sum(np.sum(image, axis=3, keepdims=True), axis=1, keepdims=True),
                axis=0,
                keepdims=True,
            ),
            axis=2,
        )
    else:
        center_x, center_y, center_z = centers

    segmentation_overlay_x = np.squeeze(image[center_x, :, :, :])
    segmentation_overlay_y = np.squeeze(image[:, center_y, :, :])
    segmentation_overlay_z = np.squeeze(image[:, :, center_z, :])

    if len(segmentation_overlay_x.shape) != 3:
        segmentation_overlay_x, segmentation_overlay_y, segmentation_overlay_z = (
            segmentation_overlay_x[..., np.newaxis],
            segmentation_overlay_y[..., np.newaxis],
            segmentation_overlay_z[..., np.newaxis],
        )

    manager.add_image(name + "_x", segmentation_overlay_x)
    manager.add_image(name + "_y", segmentation_overlay_y)
    manager.add_image(name + "_z", segmentation_overlay_z)


# todo: check through all data
def output_filename_check(torch_dataset, meta_key="image_meta_dict"):
    if len(torch_dataset) == 1:
        data = torch_dataset[0]
        if isinstance(data, list):
            data = data[0]
        return Path(data[meta_key]["filename_or_obj"]).parent.parent

    prev_data = torch_dataset[0]
    next_data = torch_dataset[1]

    if isinstance(prev_data, list):
        prev_data = prev_data[0]
    if isinstance(next_data, list):
        next_data = next_data[0]

    prev_parents = list(Path(prev_data[meta_key]["filename_or_obj"]).parents)[::-1]
    next_parents = list(Path(next_data[meta_key]["filename_or_obj"]).parents)[::-1]
    for (prev_item, next_item) in zip(prev_parents, next_parents):
        if prev_item.stem != next_item.stem:
            return prev_item.parent

    return ""


def detect_port(port):
    """Detect if the port is used"""
    socket_test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        socket_test.connect(("127.0.0.1", int(port)))
        socket_test.close()
        return True
    except:
        return False


def parse_nested_data(data):
    params = {}
    for key, value in data.items():
        if isinstance(value, dict):
            value_ = value.copy()
            if key == "lr_policy":
                policy_name = value_.get("_name", None)
                if policy_name in LR_SCHEDULES:
                    params[key] = policy_name
                    value_.pop("_name")
                    params["lr_policy_params"] = value_
            else:
                raise NotImplementedError(f"{key} is not supported for nested params.")
        else:
            params[key] = value
    return params


def is_avaible_size(value):
    if isinstance(value, (list, tuple)):
        if np.all(np.greater(value, 0)):
            return True
    return False


def get_specify_file(target_dir, regex_kw, get_first=True):
    files = list(target_dir.glob(regex_kw))
    if len(files) > 0:
        return files[0] if get_first else files
    else:
        return files


def plot_summary(summary, output_fpath):
    try:
        f = plt.figure(1)
        plt.clf()
        colors = get_colors()

        for i, (key, step_value) in enumerate(summary.items()):
            # print('Key:', key, type(key), "step_value:", step_value['values'])
            plt.plot(
                step_value["steps"],
                step_value["values"],
                label=str(key),
                color=colors[i],
                linewidth=2.0,
            )

        # plt.ylim([0., 1.])
        ax = plt.axes()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(30))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.xlabel("Number of iterations per case")
        plt.grid(True)
        plt.legend(list(summary.keys()))
        plt.draw()
        plt.show(block=False)
        plt.pause(0.0001)
        f.show()

        f.savefig(output_fpath)
    except Exception as e:
        print("Failed to do plot: " + str(e))


def dump_tensorboard(db_file, dump_keys=None, save_image=False, verbose=False):
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f"db_file is not found: {db_file}")

    if dump_keys is not None:
        dump_keys = ensure_list(dump_keys)

    def _read(input_data):
        header = struct.unpack("Q", input_data[:8])
        # crc_hdr = struct.unpack('I', input_data[:4])
        eventstr = input_data[12 : 12 + int(header[0])]  # 8+4
        out_data = input_data[12 + int(header[0]) + 4 :]
        return out_data, eventstr

    with open(db_file, "rb") as f:
        data = f.read()

    summaries = {}
    while data:
        data, event_str = _read(data)
        event = event_pb2.Event()
        event.ParseFromString(event_str)
        if event.HasField("summary"):
            for value in event.summary.value:
                if value.HasField("simple_value"):
                    if dump_keys is None or value.tag in dump_keys:
                        if not summaries.get(value.tag, None):
                            summaries[value.tag] = {
                                "steps": [event.step],
                                "values": [value.simple_value],
                            }
                        else:
                            summaries[value.tag]["steps"].append(event.step)
                            summaries[value.tag]["values"].append(value.simple_value)
                        if verbose:
                            print(value.simple_value, value.tag, event.step)
                if value.HasField("image") and save_image:
                    img = value.image
                    # save_img(img.encoded_image_string, event.step, save_gif=args.gif)
    return summaries


def _generate_color_palette(num_masks):
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    return [tuple((i * palette) % 255) for i in range(num_masks)]


def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min + 1e-5)
    return img


def norm_tensor(t, range):
    if range is not None:
        return norm_ip(t, range[0], range[1])
    else:
        return norm_ip(t, float(t.min()), float(t.max()))


def get_bound_2d(mask, connectivity=1):
    from joblib import Parallel, delayed

    if connectivity == 1:
        offset = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)]
    elif connectivity == 2:
        offset = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (0, 1, 1), (0, 1, -1), (0, -1, -1), (0, -1, 1)]
    else:
        raise ValueError(f"Connectivity should be 1 or 2, but got {connectivity}")

    def is_bound(data, coord, offsets):
        zeros = []
        for off in offsets:
            pt = tuple(coord + torch.tensor(off))
            try:
                zeros.append(data[pt] == 0)
            except:
                zeros.append(True)
        return (coord, np.any(zeros))

    boundaries = []
    if mask.ndim == 3 and mask.shape[0] > 1:
        for channel in mask[1:, ...]:
            coords = torch.nonzero(channel[None])
            if len(coords) == 0:
                continue
            bounds = Parallel(n_jobs=5)(delayed(is_bound)(channel[None], c, offset) for c in coords)
            boundary = list(filter(lambda x: x[1], bounds))
            boundaries.append(list(map(lambda x: x[0], boundary)))
    else:
        labels = np.unique(mask[mask > 0])
        for label in labels:
            coords = torch.nonzero(mask == label)
            bounds = Parallel(n_jobs=5)(delayed(is_bound)(mask, c, offset) for c in coords)
            boundary = list(filter(lambda x: x[1], bounds))
            boundaries.append(list(map(lambda x: x[0], boundary)))
    return boundaries


def __check_image_mask(image, masks):
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"The image must be a tensor, got {type(image)}")
    elif image.dtype != torch.uint8:
        raise ValueError(f"The image dtype must be uint8, got {image.dtype}")
    elif image.dim() != 3:
        raise ValueError("Pass individual images, not batches")
    elif image.size()[0] != 3:
        if image.size()[0] == 1:
            image = image.repeat_interleave(3, dim=0)
        else:
            raise ValueError(f"Pass an RGB image. Other Image formats are not supported, got {image.size()}")

    mask_value_range = masks.unique()
    if masks.ndim == 2:
        masks = masks[None, :, :]
        if len(mask_value_range) > 1:
            masks = one_hot(masks, len(mask_value_range)).type(torch.bool)
    if masks.ndim == 3:
        if len(mask_value_range) > 1 and masks.shape[0] == 1:
            masks = one_hot(masks, len(mask_value_range), dim=0).type(torch.bool)
    masks = masks.type(torch.bool)
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")

    return image, masks


def __generate_colors(colors, num_colors):
    if colors is not None and num_colors > len(colors):
        raise ValueError(f"There are more masks ({num_colors}) than colors ({len(colors)})")

    if colors is None:
        colors = _generate_color_palette(num_colors)

    if not isinstance(colors, list):
        colors = [colors]
    if not isinstance(colors[0], (tuple, str)):
        raise ValueError("colors must be a tuple or a string, or a list thereof")
    if isinstance(colors[0], tuple) and len(colors[0]) != 3:
        raise ValueError("It seems that you passed a tuple of colors instead of a list of colors")

    out_dtype = torch.uint8

    colors_ = []
    for color in colors:
        if isinstance(color, str):
            color = ImageColor.getrgb(color)
        color = torch.tensor(color, dtype=out_dtype)
        colors_.append(color)

    return colors_


def draw_segmentation_masks(
    image: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.8,
    colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
) -> torch.Tensor:

    """
    Draws segmentation masks on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        colors (list or None): List containing the colors of the masks. The colors can
            be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            When ``masks`` has a single entry of shape (H, W), you can pass a single color instead of a list
            with one element. By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """
    out_dtype = torch.uint8
    image, masks = __check_image_mask(image, masks)
    colors_ = __generate_colors(colors, masks.size()[0])

    img_to_draw = image.detach().clone()
    # TODO: There might be a way to vectorize this
    if masks.ndim == 3 and masks.shape[0] > 1:
        masks = masks[1:, ...]  # skip 0-th channel for onehotted mask
    for mask, color in zip(masks, colors_):
        img_to_draw[:, mask] = color[:, None]

    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)


def draw_segmentation_contour(
    image: torch.Tensor,
    masks: torch.Tensor,
    alpha: float = 0.8,
    radius: float = 0.2,
    colors: Optional[List[Union[str, Tuple[int, int, int]]]] = None,
):
    """
    Draws segmentation contour on given RGB image.
    The values of the input image should be uint8 between 0 and 255.

    Args:
        image (Tensor): Tensor of shape (3, H, W) and dtype uint8.
        masks (Tensor): Tensor of shape (num_masks, H, W) or (H, W) and dtype bool.
        alpha (float): Float number between 0 and 1 denoting the transparency of the masks.
            0 means full transparency, 1 means no transparency.
        radius (int): Integer denoting radius of keypoint.
        colors (list or None): List containing the colors of the masks. The colors can
            be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``.
            When ``masks`` has a single entry of shape (H, W), you can pass a single color instead of a list
            with one element. By default, random colors are generated for each mask.

    Returns:
        img (Tensor[C, H, W]): Image Tensor, with segmentation masks drawn on top.
    """
    out_dtype = torch.uint8
    image, masks = __check_image_mask(image, masks)
    colors_ = __generate_colors(colors, masks.size()[0])

    ndarr = image.permute(1, 2, 0).numpy()
    img_to_draw = Image.fromarray(ndarr)
    draw = ImageDraw.Draw(img_to_draw)
    boundaries = get_bound_2d(masks)

    for i, bound in enumerate(boundaries):
        for inst_id, pt in enumerate(bound):
            x1, x2 = pt[2] - radius, pt[2] + radius  # because permute
            y1, y2 = pt[1] - radius, pt[1] + radius
            draw.ellipse([x1, y1, x2, y2], fill=tuple(colors_[i]), outline=None, width=0)

    return torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)


class LogColorFormatter(logging.Formatter):
    """Logging colored formatter, adapted from https://stackoverflow.com/a/56944256/3638629"""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt=None):
        super().__init__()
        self.fmt = fmt if fmt else "%(asctime)s %(name)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)"
        self.FORMATS = {
            logging.DEBUG: self.grey + self.fmt + self.reset,
            logging.INFO: self.blue + self.fmt + self.reset,
            logging.WARNING: self.yellow + self.fmt + self.reset,
            logging.ERROR: self.red + self.fmt + self.reset,
            logging.CRITICAL: self.bold_red + self.fmt + self.reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def setup_logger(
    name: Optional[str] = "ignite",
    level: int = logging.INFO,
    stream: Optional[TextIO] = None,
    format: str = "%(asctime)s %(name)s %(levelname)s: %(message)s (%(filename)s:%(lineno)d)",
    color: bool = True,
    filepath: Optional[str] = None,
    distributed_rank: Optional[int] = None,
    reset: bool = False,
    terminator: str = "\n",
) -> logging.Logger:
    """
    Extented ignite's setup_logger.
    Extended: `color`, `terminator`.

    Setups logger: name, level, format etc.

    Args:
        name: new name for the logger. If None, the standard logger is used.
        level: logging level, e.g. CRITICAL, ERROR, WARNING, INFO, DEBUG.
        stream: logging stream. If None, the standard stream is used (sys.stderr).
        format: logging format. By default, `%(asctime)s %(name)s %(levelname)s: %(message)s`.
        color: whether use colored log
        filepath: Optional logging file path. If not None, logs are written to the file.
        distributed_rank: Optional, rank in distributed configuration to avoid logger setup for workers.
            If None, distributed_rank is initialized to the rank of process.
        reset: if True, reset an existing logger rather than keep format, handlers, and level.
        terminator: change the terminator of output stream. eg. `\r` for concise msg

    Returns:
        logging.Logger

    Examples:
        Improve logs readability when training with a trainer and evaluator:

        .. code-block:: python

            from ignite.utils import setup_logger

            trainer = ...
            evaluator = ...

            trainer.logger = setup_logger("trainer")
            evaluator.logger = setup_logger("evaluator")

            trainer.run(data, max_epochs=10)

            # Logs will look like
            # 2020-01-21 12:46:07,356 trainer INFO: Engine run starting with max_epochs=5.
            # 2020-01-21 12:46:07,358 trainer INFO: Epoch[1] Complete. Time taken: 00:5:23
            # 2020-01-21 12:46:07,358 evaluator INFO: Engine run starting with max_epochs=1.
            # 2020-01-21 12:46:07,358 evaluator INFO: Epoch[1] Complete. Time taken: 00:01:02
            # ...

        Every existing logger can be reset if needed

        .. code-block:: python

            logger = setup_logger(name="my-logger", format="=== %(name)s %(message)s")
            logger.info("first message")
            setup_logger(name="my-logger", format="+++ %(name)s %(message)s", reset=True)
            logger.info("second message")

            # Logs will look like
            # === my-logger first message
            # +++ my-logger second message

        Change the level of an existing internal logger

        .. code-block:: python

            setup_logger(
                name="ignite.distributed.launcher.Parallel",
                level=logging.WARNING
            )
    """
    # check if the logger already exists
    existing = name is None or name in logging.root.manager.loggerDict

    # if existing, get the logger otherwise create a new one
    logger = logging.getLogger(name)

    if distributed_rank is None:
        import ignite.distributed as idist

        distributed_rank = idist.get_rank()

    # Remove previous handlers
    if distributed_rank > 0 or reset:

        if logger.hasHandlers():
            for h in list(logger.handlers):
                logger.removeHandler(h)

    if distributed_rank > 0:

        # Add null handler to avoid multiple parallel messages
        logger.addHandler(logging.NullHandler())

    # Keep the existing configuration if not reset
    if existing and not reset:
        return logger

    if distributed_rank == 0:
        logger.setLevel(level)

        formatter = LogColorFormatter(format) if color else logging.Formatter(format)

        ch = logging.StreamHandler(stream=stream)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        ch.terminator = terminator
        logger.addHandler(ch)

        if filepath is not None:
            fh = logging.FileHandler(filepath)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(format))  # file no color
            logger.addHandler(fh)

    # don't propagate to ancestors
    # the problem here is to attach handlers to loggers
    # should we provide a default configuration less open ?
    if name is not None:
        logger.propagate = False

    return logger


def warning_on_one_line(message, category, filename, lineno, file=None, line=None):
    return '%s:%s:\n %s: %s\n' % (filename, lineno, category.__name__, message)


def singleton(cls):
    _instance = {}

    def _singleton(*args, **kwargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return _singleton


def save_sourcecode(code_rootdir, out_dir, verbose=True):
    if not os.path.isdir(code_rootdir):
        raise FileNotFoundError(f"Code root dir not exists! {code_rootdir}")

    Print("Backup source code under root_dir:", code_rootdir, color="y", verbose=verbose)
    outpath = out_dir / f"{os.path.basename(code_rootdir)}_{time.strftime('%m%d_%H%M')}.tar"
    tar_opt = "cvf" if verbose else "cf"
    os.system(f"cd {str(code_rootdir)}; tar -{tar_opt} {outpath} .")


def get_torch_datast(strix_dataset, phase: Phases, opts: dict, synthetic_data_num=100, split_func: Optional[callable] = None):
    """This function return pytorch dataset generated by registered strix dataset.

    Args:
        strix_dataset (dict): strix dastset got from DatasetRegistry
        phase (Phases): specific phase
        opts (dict): hyper-parameters pass to dataset
        synthetic_data_num (int): if `strix_dataset["PATH"]` is not given, generate some dummy file path
        split_func (callable): function provided to split data list
    Returns:
        dict: pytorch dataset
    """
    try:
        dataset_fn, dataset_list = strix_dataset["FN"], strix_dataset["PATH"]
        if dataset_list:
            filelist = get_items(dataset_list, format="auto")
        else:
            filelist = generate_synthetic_datalist(synthetic_data_num)

        if split_func is not None:
            train_flist, valid_flist = split_func(filelist)
            if phase == Phases.TRAIN:
                filelist = train_flist
            elif phase == Phases.VALID:
                filelist = valid_flist
            elif phase == Phases.TEST_EX or phase == Phases.TEST_IN:
                filelist = strix_dataset.get("TEST_PATH")
                if not filelist:
                    raise FileNotFoundError("No test path is specified during registering")
        torch_dataset = dataset_fn(filelist, phase, opts)
    except KeyError as e:
        warnings.warn(colored(f"Dataset registered error!\nErr msg: {repr(e)}\n{e.__cause__}", "red"))
        return None
    except Exception as e:
        warnings.warn(colored(f"Creating dataset '{strix_dataset}' failed! \nMsg: {repr(e)}", "red"))
        return None
    else:
        return torch_dataset
