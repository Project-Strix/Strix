from typing import Any

import os
import torch

from strix import strix_networks
from strix.models.cnn.nets.resnet import resnet18, resnet34, resnet50
from strix.models.cnn.nets.vgg import vgg9_bn, vgg11_bn
from strix.models.cnn.nets.dynunet import DynUNet
from strix.models.cnn.nets.drn import drn_a_50
from strix.models.cnn.nets.resnet_aag import resnet34_aag, resnet50_aag


@strix_networks.register("2D", "classification", "resnet18")
@strix_networks.register("3D", "classification", "resnet18")
def strix_resnet18(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["dim"] = spatial_dims
    inkwargs["in_channels"] = in_channels
    inkwargs["num_classes"] = out_channels
    inkwargs["groups"] = n_group

    return resnet18(pretrained=False, progress=True, **inkwargs)


@strix_networks.register("2D", "classification", "resnet34")
@strix_networks.register("3D", "classification", "resnet34")
def strix_resnet50(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["dim"] = spatial_dims
    inkwargs["in_channels"] = in_channels
    inkwargs["num_classes"] = out_channels
    inkwargs["groups"] = n_group

    return resnet34(pretrained=False, progress=True, **inkwargs)


@strix_networks.register("2D", "classification", "resnet50")
@strix_networks.register("3D", "classification", "resnet50")
def strix_resnet50(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["dim"] = spatial_dims
    inkwargs["in_channels"] = in_channels
    inkwargs["num_classes"] = out_channels
    inkwargs["groups"] = n_group

    return resnet50(pretrained=False, progress=True, **inkwargs)


@strix_networks.register("2D", "classification", "vgg9")
@strix_networks.register("3D", "classification", "vgg9")
def strix_vgg9_bn(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["dim"] = spatial_dims
    inkwargs["in_channels"] = in_channels
    inkwargs["num_classes"] = out_channels
    inkwargs["is_prunable"] = is_prunable
    inkwargs["groups"] = n_group
    inkwargs["bottleneck_size"] = kwargs.get("bottleneck_size", 5)

    return vgg9_bn(pretrained=False, progress=True, **inkwargs)


@strix_networks.register("2D", "classification", "vgg11")
@strix_networks.register("3D", "classification", "vgg11")
def strix_vgg11_bn(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["dim"] = spatial_dims
    inkwargs["in_channels"] = in_channels
    inkwargs["num_classes"] = out_channels
    inkwargs["is_prunable"] = is_prunable
    inkwargs["groups"] = n_group
    inkwargs["bottleneck_size"] = kwargs.get("bottleneck_size", 2)

    return vgg11_bn(pretrained=False, progress=True, **inkwargs)


@strix_networks.register('2D', "selflearning", "unet")
@strix_networks.register('3D', "selflearning", "unet")
def strix_dyn_unet(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    n_depth = 5 if n_depth == -1 else n_depth
    kernel_size = kwargs.get("kernel_size", (3,) + (3,) * n_depth)
    strides = kwargs.get("strides", (1,) + (2,) * n_depth)
    upsample_kernel_size = kwargs.get("upsample_kernel_size", (1,) + (2,) * n_depth)
    deep_supervision = kwargs.get("deep_supervision", False)
    deep_supr_num = kwargs.get("deep_supr_num", 1)
    res_block = False
    last_activation = "sigmoid"
    filters = kwargs.get("filters", None)
    output_bottleneck = kwargs.get("output_bottleneck", False)

    return DynUNet(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        strides,
        upsample_kernel_size,
        norm,
        deep_supervision,
        deep_supr_num,
        res_block,
        last_activation,
        is_prunable,
        filters,
        output_bottleneck,
    )


@strix_networks.register("2D", "segmentation", "unet")
@strix_networks.register("3D", "segmentation", "unet")
def strix_dyn_unet(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    n_depth = 5 if n_depth == -1 else n_depth
    kernel_size = kwargs.get("kernel_size", (3,) + (3,) * n_depth)
    strides = kwargs.get("strides", (1,) + (2,) * n_depth)
    upsample_kernel_size = kwargs.get("upsample_kernel_size", (1,) + (2,) * n_depth)
    deep_supervision = kwargs.get("deep_supervision", False)
    deep_supr_num = kwargs.get("deep_supr_num", 1)
    res_block = False
    last_activation = None
    filters = kwargs.get("filters", None)
    output_bottleneck = kwargs.get("output_bottleneck", False)

    return DynUNet(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        strides,
        upsample_kernel_size,
        norm,
        deep_supervision,
        deep_supr_num,
        res_block,
        last_activation,
        is_prunable,
        filters,
        output_bottleneck,
    )


@strix_networks.register("2D", "segmentation", "res-unet")
@strix_networks.register("3D", "segmentation", "res-unet")
def strix_dyn_resunet(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    n_depth = 5 if n_depth == -1 else n_depth
    kernel_size = kwargs.get("kernel_size", (3,) + (3,) * n_depth)
    strides = kwargs.get("strides", (1,) + (2,) * n_depth)
    upsample_kernel_size = kwargs.get("upsample_kernel_size", (1,) + (2,) * n_depth)
    deep_supervision = kwargs.get("deep_supervision", False)
    deep_supr_num = kwargs.get("deep_supr_num", 1)
    res_block = True
    last_activation = None
    filters = kwargs.get("filters", None)
    output_bottleneck = kwargs.get("output_bottleneck", False)

    return DynUNet(
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        strides,
        upsample_kernel_size,
        norm,
        deep_supervision,
        deep_supr_num,
        res_block,
        last_activation,
        is_prunable,
        filters,
        output_bottleneck,
    )


@strix_networks.register("2D", "classification", "DRN_a50")
@strix_networks.register("3D", "classification", "DRN_a50")
def strix_drn_a_50(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    inkwargs = {}
    inkwargs["in_channels"] = in_channels
    inkwargs["num_classes"] = out_channels

    return drn_a_50(pretrained=False, **inkwargs)


@strix_networks.register("2D", "classification", "resnet_aag_34")
@strix_networks.register("3D", "classification", "resnet_aag_34")
def strix_resnetaag_34(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    r"""ResNetAAG-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inkwargs = {}
    inkwargs["dim"] = spatial_dims
    inkwargs["in_channels"] = in_channels
    inkwargs["num_classes"] = out_channels
    inkwargs["groups"] = n_group
    # inkwargs["pretrained_model_path"] = pretrained_model_path
    inkwargs["roi_classes"] = int(kwargs.get("roi_classes", 3))
    inkwargs["freeze_backbone"] = kwargs.get("freeze_backbone", False)
    if inkwargs["freeze_backbone"]:
        print("Freeze backbone for fine-tune!")

    return resnet34_aag(pretrained_model_path, **inkwargs)


@strix_networks.register("2D", "classification", "resnet_aag_50")
@strix_networks.register("3D", "classification", "resnet_aag_50")
def strix_resnetaag_50(
    spatial_dims: int,
    in_channels: int,
    out_channels: int,
    act: str,
    norm: str,
    n_depth: int,
    n_group: int,
    drop_out: float,
    is_prunable: bool,
    pretrained: bool,
    pretrained_model_path: str,
    **kwargs: Any
):
    r"""ResNetAAG-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    inkwargs = {}
    inkwargs["dim"] = spatial_dims
    inkwargs["in_channels"] = in_channels
    inkwargs["num_classes"] = out_channels
    inkwargs["groups"] = n_group
    # inkwargs["pretrained_model_path"] = pretrained_model_path
    inkwargs["roi_classes"] = int(kwargs.get("roi_classes", 3))
    inkwargs["freeze_backbone"] = kwargs.get("freeze_backbone", False)
    if inkwargs["freeze_backbone"]:
        print("Freeze backbone for fine-tune!")

    return resnet50_aag(pretrained_model_path, **inkwargs)
