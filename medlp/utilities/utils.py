from __future__ import print_function
from typing import Sequence, Union

import os, math, random, copy, pylab, torch
from pathlib import Path
import socket
from medlp.utilities.enum import NETWORK_TYPES, DIMS
import numpy as np

from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.cm as mpl_color_map


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
    assert len(torch_dataset) > 1, 'dataset must have at least 2 items!'
    prev_data = torch_dataset[0]
    next_data = torch_dataset[1]

    if Path(prev_data[meta_key]['filename_or_obj']).stem != Path(next_data[meta_key]['filename_or_obj']).stem:
        return 0

    for i, (prev_v, next_v) in enumerate(zip(Path(prev_data[meta_key]['filename_or_obj']).parents,
                                             Path(next_data[meta_key]['filename_or_obj']).parents)):
        if prev_v.stem != next_v.stem:
            return i+1

    return 0

def detect_port(port):
    '''Detect if the port is used'''
    socket_test = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        socket_test.connect(('127.0.0.1', int(port)))
        socket_test.close()
        return True
    except:
        return False

def get_network_type(name):
    for k, v in NETWORK_TYPES.items():
        if name in v:
            return k

def assert_network_type(model_name, target_type):
    assert get_network_type(model_name) == target_type, f"Only accept {target_type} arch: {NETWORK_TYPES[target_type]}"

def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module

def _register_generic_dim(module_dict, dim, module_name, module):
    assert module_name not in module_dict.get(dim), f'{module_name} already registed in {module_dict.get(dim)}'
    module_dict[dim].update({module_name:module})

def _register_generic_data(module_dict, dim, module_name, fpath, module):
    assert module_name not in module_dict.get(dim), f'{module_name} already registed in {module_dict.get(dim)}'
    module_dict[dim].update({module_name:module, module_name+"_fpath":fpath})


def is_avaible_size(value):
    if isinstance(value, (list, tuple)):
        if np.all(np.greater(value, 0)):
            return True
    return False

def classification_scores(gts, preds, labels):
    accuracy        = metrics.accuracy_score(gts,  preds)
    class_accuracies = []
    for lab in labels: # TODO Fix
        class_accuracies.append(metrics.accuracy_score(gts[gts == lab], preds[gts == lab]))
    class_accuracies = np.array(class_accuracies)

    f1_micro        = metrics.f1_score(gts,        preds, average='micro')
    precision_micro = metrics.precision_score(gts, preds, average='micro')
    recall_micro    = metrics.recall_score(gts,    preds, average='micro')
    f1_macro        = metrics.f1_score(gts,        preds, average='macro')
    precision_macro = metrics.precision_score(gts, preds, average='macro')
    recall_macro    = metrics.recall_score(gts,    preds, average='macro')

    # class wise score
    f1s        = metrics.f1_score(gts,        preds, average=None)
    precisions = metrics.precision_score(gts, preds, average=None)
    recalls    = metrics.recall_score(gts,    preds, average=None)

    confusion = metrics.confusion_matrix(gts,preds, labels=labels)

    #TODO confusion matrix, recall, precision
    return accuracy, f1_micro, precision_micro, recall_micro, f1_macro, precision_macro, recall_macro, confusion, class_accuracies, f1s, precisions, recalls


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.

    Eg. creeting a registry:
        some_registry = Registry({"default": default_module})

    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...

    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn

class DimRegistry(dict):
    def __init__(self, *args, **kwargs):
        super(DimRegistry, self).__init__(*args, **kwargs)
        self.dim_mapping = {'2': '2D', '3': '3D', 2: '2D', 3: '3D', '2D': '2D', '3D': '3D'}
        self['2D'] = {}
        self['3D'] = {}

    def register(self, dim, module_name, module=None):
        assert dim in DIMS, "Only support 2D&3D dataset now"
        dim = self.dim_mapping[dim]
        # used as function call
        if module is not None:
            _register_generic_dim(self, dim, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic_dim(self, dim, module_name, fn)
            return fn 
        return register_fn


class DatasetRegistry(DimRegistry):
    def __init__(self, *args, **kwargs):
        super(DatasetRegistry, self).__init__(*args, **kwargs)

    def register(self, dim, module_name, fpath, module=None):
        assert dim in DIMS, "Only support 2D&3D dataset now"
        dim = self.dim_mapping[dim]
        # used as function call
        if module is not None:
            _register_generic_data(self, dim, module_name, fpath, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic_data(self, dim, module_name, fpath, fn)
            return fn 
        return register_fn


