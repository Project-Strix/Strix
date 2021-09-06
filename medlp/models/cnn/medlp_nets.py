from typing import Any

import os
import torch

from medlp.models.cnn import CLASSIFICATION_ARCHI, SEGMENTATION_ARCHI
from medlp.models.cnn.nets.resnet import resnet18, resnet34, resnet50
from medlp.models.cnn.nets.vgg import vgg9_bn
from medlp.models.cnn.nets.dynunet import DynUNet
from medlp.models.cnn.nets.drn import drn_a_50
from medlp.models.cnn.nets.hesam import HESAM
from medlp.models.cnn.nets.resnet_aag import resnet34_aag


@CLASSIFICATION_ARCHI.register('2D', 'resnet18')
@CLASSIFICATION_ARCHI.register('3D', 'resnet18')
def medlp_resnet18(
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
    kwargs['dim'] = spatial_dims
    kwargs['in_channels'] = in_channels
    kwargs['num_classes'] = out_channels
    kwargs['groups'] = n_group

    return resnet18(pretrained=False, progress=True, **kwargs)


@CLASSIFICATION_ARCHI.register('2D', 'resnet34')
@CLASSIFICATION_ARCHI.register('3D', 'resnet34')
def medlp_resnet50(
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
    kwargs['dim'] = spatial_dims
    kwargs['in_channels'] = in_channels
    kwargs['num_classes'] = out_channels
    kwargs['groups'] = n_group

    return resnet34(pretrained=False, progress=True, **kwargs)


@CLASSIFICATION_ARCHI.register('2D', 'resnet50')
@CLASSIFICATION_ARCHI.register('3D', 'resnet50')
def medlp_resnet50(
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
    kwargs['dim'] = spatial_dims
    kwargs['in_channels'] = in_channels
    kwargs['num_classes'] = out_channels
    kwargs['groups'] = n_group

    return resnet50(pretrained=False, progress=True, **kwargs)


@CLASSIFICATION_ARCHI.register('2D', 'vgg9_bn')
@CLASSIFICATION_ARCHI.register('3D', 'vgg9_bn')
def medlp_vgg9_bn(
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
    kwargs['dim'] = spatial_dims
    kwargs['in_channels'] = in_channels
    kwargs['num_classes'] = out_channels
    kwargs['is_prunable'] = is_prunable
    kwargs['groups'] = n_group

    return vgg9_bn(pretrained=False, progress=True, **kwargs)


# @SELFLEARNING_ARCHI.register('2D', 'unet')
# @SELFLEARNING_ARCHI.register('3D', 'unet')
@SEGMENTATION_ARCHI.register('2D', 'unet')
@SEGMENTATION_ARCHI.register('3D', 'unet')
def medlp_dyn_unet(
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
    kernel_size = kwargs.get("kernel_size", (3,)+(3,)*n_depth)
    strides = kwargs.get("strides", (1,)+(2,)*n_depth)
    upsample_kernel_size = kwargs.get("upsample_kernel_size", (1,)+(2,)*n_depth)
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
        output_bottleneck
    )


@SEGMENTATION_ARCHI.register('2D', 'res-unet')
@SEGMENTATION_ARCHI.register('3D', 'res-unet')
def medlp_dyn_unet(
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
    kernel_size = kwargs.get("kernel_size", (3,)+(3,)*n_depth)
    strides = kwargs.get("strides", (1,)+(2,)*n_depth)
    upsample_kernel_size = kwargs.get("upsample_kernel_size", (1,)+(2,)*n_depth)
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
        output_bottleneck
    )


@CLASSIFICATION_ARCHI.register('2D', 'DRN_a50')
@CLASSIFICATION_ARCHI.register('3D', 'DRN_a50')
def medlp_drn_a_50(
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
    kwargs['in_channels'] = in_channels
    kwargs['num_classes'] = out_channels

    return drn_a_50(pretrained=False, **kwargs)


@CLASSIFICATION_ARCHI.register('2D', 'HESAM')
@CLASSIFICATION_ARCHI.register('3D', 'HESAM')
def medlp_hesam(
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

    net = HESAM(
        spatial_dims,
        in_channels,
        out_channels,
        features,
        last_feature,
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


@CLASSIFICATION_ARCHI.register('2D', 'resnet_aag_34')
@CLASSIFICATION_ARCHI.register('3D', 'resnet_aag_34')
def medlp_resnetaag_34(
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
    inkwargs['dim'] = spatial_dims
    inkwargs['in_channels'] = in_channels
    inkwargs['num_classes'] = out_channels
    inkwargs['groups'] = n_group
    inkwargs["roi_classes"] = int(kwargs.get("roi_classes", 3))

    return resnet34_aag(pretrained, **inkwargs)
