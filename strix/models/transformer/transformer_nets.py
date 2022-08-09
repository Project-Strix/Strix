from typing import Any

from strix.utilities.registry import NetworkRegistry
from monai_ex.networks.nets import UNETR

NETWORK = NetworkRegistry()

@NETWORK.register("2D", "segmentation", "UNETR")
@NETWORK.register("3D", "segmentation", "UNETR")
def strix_unetr(
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
    image_sz = kwargs.get("image_size")
    pos_embed = kwargs.get("pos_embed", "perceptron")  # conv
    num_heads = kwargs.get("num_heads", 12)
    feature_size = kwargs.get("feature_size", 16)
    if image_sz is None:
        raise ValueError("Please specify input image size!")

    return UNETR(
        spatial_dims=spatial_dims,
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=image_sz,
        feature_size=int(feature_size),
        hidden_size=768,
        mlp_dim=3072,
        num_heads=int(num_heads),
        pos_embed=pos_embed,
        norm_name=norm,
        conv_block=True,
        res_block=True,
    )
