from __future__ import print_function
from typing import Union, Optional, List, Tuple

import os
import struct
import pylab
import torch
from pathlib import Path
from PIL import Image, ImageColor
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mcolors
import numpy as np

import matplotlib.cm as mpl_color_map

from medlp.utilities.enum import LR_SCHEDULE
from monai_ex.utils import ensure_list
import tensorboard.compat.proto.event_pb2 as event_pb2


def get_attr_(obj, name, default):
    return getattr(obj, name) if hasattr(obj, name) else default


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
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def create_rgb_summary(label):
    num_colors = label.shape[1]

    cm = pylab.get_cmap('gist_rainbow')

    new_label = np.zeros((label.shape[0], label.shape[1], label.shape[2], 3), dtype=np.float32)

    for i in range(num_colors):
        color = cm(1. * i / num_colors)  # color will now be an RGBA tuple
        new_label[:, :, :, 0] += label[:, :, :, i] * color[0]
        new_label[:, :, :, 1] += label[:, :, :, i] * color[1]
        new_label[:, :, :, 2] += label[:, :, :, i] * color[2]

    return new_label


def add_3D_overlay_to_summary(
    data: Union[torch.Tensor, np.ndarray],
    mask: Union[torch.Tensor, np.ndarray],
    writer,
    index: int = 0,
    tag: str = 'output',
    centers=None
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
        center_x = np.argmax(np.sum(np.sum(np.sum(mask_, axis=3, keepdims=True), axis=2, keepdims=True), axis=1, keepdims=True), axis=0)
        center_y = np.argmax(np.sum(np.sum(np.sum(mask_, axis=3, keepdims=True), axis=2, keepdims=True), axis=0, keepdims=True), axis=1)
        center_z = np.argmax(np.sum(np.sum(np.sum(mask_, axis=3, keepdims=True), axis=1, keepdims=True), axis=0, keepdims=True), axis=2)
    else:
        center_x, center_y, center_z = centers

    segmentation_overlay_x = \
        np.squeeze(data_[center_x, :, :, :] + mask_[center_x, :, :, :])
    segmentation_overlay_y = \
        np.squeeze(data_[:, center_y, :, :] + mask_[:, center_y, :, :])
    segmentation_overlay_z = \
        np.squeeze(data_[:, :, center_z, :] + mask_[:, :, center_z, :])

    if len(segmentation_overlay_x.shape) != 3:
        segmentation_overlay_x, segmentation_overlay_y, segmentation_overlay_z = \
            segmentation_overlay_x[..., np.newaxis], \
            segmentation_overlay_y[..., np.newaxis], \
            segmentation_overlay_z[..., np.newaxis]

    writer.add_image(tag + '_x', segmentation_overlay_x)
    writer.add_image(tag + '_y', segmentation_overlay_y)
    writer.add_image(tag + '_z', segmentation_overlay_z)


def add_3D_image_to_summary(manager, image, name, centers=None):
    image = np.squeeze(image)

    if len(image.shape) > 3:
        # there are channels
        print('add_3D_image_to_summary: there are channels')
        image = create_rgb_summary(image)
    else:
        image = image[..., np.newaxis]

    if centers is None:
        center_x = np.argmax(np.sum(np.sum(np.sum(image, axis=3, keepdims=True), axis=2, keepdims=True), axis=1, keepdims=True), axis=0)
        center_y = np.argmax(np.sum(np.sum(np.sum(image, axis=3, keepdims=True), axis=2, keepdims=True), axis=0, keepdims=True), axis=1)
        center_z = np.argmax(np.sum(np.sum(np.sum(image, axis=3, keepdims=True), axis=1, keepdims=True), axis=0, keepdims=True), axis=2)
    else:
        center_x, center_y, center_z = centers

    segmentation_overlay_x = np.squeeze(image[center_x, :, :, :])
    segmentation_overlay_y = np.squeeze(image[:, center_y, :, :])
    segmentation_overlay_z = np.squeeze(image[:, :, center_z, :])

    if len(segmentation_overlay_x.shape) != 3:
        segmentation_overlay_x, segmentation_overlay_y, segmentation_overlay_z = \
            segmentation_overlay_x[..., np.newaxis], \
            segmentation_overlay_y[..., np.newaxis], \
            segmentation_overlay_z[..., np.newaxis]

    manager.add_image(name + '_x', segmentation_overlay_x)
    manager.add_image(name + '_y', segmentation_overlay_y)
    manager.add_image(name + '_z', segmentation_overlay_z)


def output_filename_check(torch_dataset, meta_key='image_meta_dict'):
    if len(torch_dataset) == 1:
        return Path(torch_dataset[0][meta_key]['filename_or_obj']).parent.parent

    prev_data = torch_dataset[0]
    next_data = torch_dataset[1]

    if Path(prev_data[meta_key]['filename_or_obj']).stem != Path(next_data[meta_key]['filename_or_obj']).stem:
        return Path(prev_data[meta_key]['filename_or_obj']).parent

    for i, (prev_v, next_v) in enumerate(zip(Path(prev_data[meta_key]['filename_or_obj']).parents,
                                             Path(next_data[meta_key]['filename_or_obj']).parents)):
        if prev_v.stem != next_v.stem:
            return prev_v.parent

    return ''


def detect_port(port):
    '''Detect if the port is used'''
    socket_test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        socket_test.connect(('127.0.0.1', int(port)))
        socket_test.close()
        return True
    except:
        return False


def parse_nested_data(data):
    params = {}
    for key, value in data.items():
        if isinstance(value, dict):
            value_ = value.copy()
            if key == 'lr_policy':
                policy_name = value_.get('_name', None)
                if policy_name in LR_SCHEDULE:
                    params[key] = policy_name
                    value_.pop('_name')
                    params['lr_policy_params'] = value_
            else:
                raise NotImplementedError(f'{key} is not supported for nested params.')
        else:
            params[key] = value
    return params


def is_avaible_size(value):
    if isinstance(value, (list, tuple)):
        if np.all(np.greater(value, 0)):
            return True
    return False


def plot_summary(summary, output_fpath):
    try:
        f = plt.figure(1)
        plt.clf()
        colors = list(mcolors.TABLEAU_COLORS.values())

        for i, (key, step_value) in enumerate(summary.items()):
            plt.plot(step_value['steps'], step_value['values'], label=str(key), color=colors[i], linewidth=2.0)
        # plt.ylim([0., 1.])
        ax = plt.axes()
        ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.xlabel('Number of iterations per case')
        plt.grid(True)
        ax.legend()
        plt.draw()
        plt.show(block=False)
        plt.pause(0.0001)
        f.show()

        f.savefig(output_fpath)
    except Exception as e:
        print('Failed to do plot: ' + str(e))


def dump_tensorboard(db_file, dump_keys=None, save_image=False, verbose=False):
    if not os.path.isfile(db_file):
        raise FileNotFoundError(f'db_file is not found: {db_file}')

    if dump_keys is not None:
        dump_keys = ensure_list(dump_keys)

    def _read(input_data):
        header = struct.unpack('Q', input_data[:8])
        # crc_hdr = struct.unpack('I', input_data[:4])
        eventstr = input_data[12:12+int(header[0])]  # 8+4
        out_data = input_data[12+int(header[0])+4:]
        return out_data, eventstr

    with open(db_file, 'rb') as f:
        data = f.read()

    summaries = {}
    while data:
        data, event_str = _read(data)
        event = event_pb2.Event()
        event.ParseFromString(event_str)
        if event.HasField('summary'):
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    if dump_keys is None or value.tag in dump_keys:
                        if not summaries.get(value.tag, None):
                            summaries[value.tag] = {'steps': [event.step], 'values': [value.simple_value]}
                        else:
                            summaries[value.tag]['steps'].append(event.step)
                            summaries[value.tag]['values'].append(value.simple_value)
                        if verbose:
                            print(value.simple_value, value.tag, event.step)
                if value.HasField('image') and save_image:
                    img = value.image
                    # save_img(img.encoded_image_string, event.step, save_gif=args.gif)
    return summaries


def _generate_color_palette(num_masks):
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
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
    if masks.ndim == 2:
        masks = masks[None, :, :]
    if masks.ndim != 3:
        raise ValueError("masks must be of shape (H, W) or (batch_size, H, W)")
    if masks.dtype != torch.bool:
        raise ValueError(f"The masks must be of dtype bool. Got {masks.dtype}")
    if masks.shape[-2:] != image.shape[-2:]:
        raise ValueError("The image and the masks must have the same height and width")

    num_masks = masks.size()[0]
    if colors is not None and num_masks > len(colors):
        raise ValueError(f"There are more masks ({num_masks}) than colors ({len(colors)})")

    if colors is None:
        colors = _generate_color_palette(num_masks)

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

    img_to_draw = image.detach().clone()
    # TODO: There might be a way to vectorize this
    for mask, color in zip(masks, colors_):
        img_to_draw[:, mask] = color[:, None]

    out = image * (1 - alpha) + img_to_draw * alpha
    return out.to(out_dtype)


