from typing import Any

import os
import torch

from strix.models import CLASSIFICATION_ARCHI, SEGMENTATION_ARCHI, SELFLEARNING_ARCHI
from strix.models.cnn.nets.resnet import resnet18, resnet34, resnet50
from strix.models.cnn.nets.vgg import vgg9_bn, vgg11_bn
from strix.models.cnn.nets.dynunet import DynUNet
from strix.models.cnn.nets.drn import drn_a_50
from strix.models.cnn.nets.hesam import HESAM
from strix.models.cnn.nets.resnet_aag import resnet34_aag, resnet50_aag


@CLASSIFICATION_ARCHI.register("2D", "resnet18")
@CLASSIFICATION_ARCHI.register("3D", "resnet18")
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


@CLASSIFICATION_ARCHI.register("2D", "resnet34")
@CLASSIFICATION_ARCHI.register("3D", "resnet34")
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


@CLASSIFICATION_ARCHI.register("2D", "resnet50")
@CLASSIFICATION_ARCHI.register("3D", "resnet50")
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


@CLASSIFICATION_ARCHI.register("2D", "vgg9")
@CLASSIFICATION_ARCHI.register("3D", "vgg9")
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


@CLASSIFICATION_ARCHI.register("2D", "vgg11")
@CLASSIFICATION_ARCHI.register("3D", "vgg11")
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


@SELFLEARNING_ARCHI.register('2D', 'unet')
@SELFLEARNING_ARCHI.register('3D', 'unet')
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


@SEGMENTATION_ARCHI.register("2D", "unet")
@SEGMENTATION_ARCHI.register("3D", "unet")
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


@SEGMENTATION_ARCHI.register("2D", "res-unet")
@SEGMENTATION_ARCHI.register("3D", "res-unet")
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


@CLASSIFICATION_ARCHI.register("2D", "DRN_a50")
@CLASSIFICATION_ARCHI.register("3D", "DRN_a50")
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


@CLASSIFICATION_ARCHI.register("2D", "HESAM")
@CLASSIFICATION_ARCHI.register("3D", "HESAM")
def strix_hesam(
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
    features = kwargs.get("features", (64, 128, 256, 256))
    last_feature = kwargs.get("last_feature", 64)
    upsample = kwargs.get("upsample", "deconv")
    sam_size = kwargs.get("sam_size", 6)

    net = HESAM(
        spatial_dims,
        in_channels,
        out_channels,
        features,
        last_feature,
        sam_size,
        act,
        norm,
        drop_out,
        upsample,
        n_group,
    )

    if os.path.isfile(pretrained_model_path):
        print("Load pretrained model for contiune training:\n")
        net.load_state_dict(torch.load(pretrained_model_path))

        fc = torch.nn.Linear(features[-1], out_channels)
        net.final_fc = fc

    return net


@CLASSIFICATION_ARCHI.register("2D", "resnet_aag_34")
@CLASSIFICATION_ARCHI.register("3D", "resnet_aag_34")
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


@CLASSIFICATION_ARCHI.register("2D", "resnet_aag_50")
@CLASSIFICATION_ARCHI.register("3D", "resnet_aag_50")
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
