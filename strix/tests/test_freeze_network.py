import pytest
import warnings 

import torch
import torch.nn as nn

from strix import strix_networks
from strix.models.cnn.utils import count_trainable_params


@pytest.mark.parametrize("freeze", [True, False])
def test_freeze_network(freeze):
    network = strix_networks.get("3D", "multitask", "UnifiedNet-M3")
    if network is None:
        warnings.warn("UnifiedNet-M3 not registered! Please use other multitask network to test! Skip")
        return

    net = network(
        3,
        1,
        (1, 1),
        act='relu',
        norm='batch',
        n_depth=5,
        n_group=1,
        drop_out=0,
        is_prunable=False,
        pretrained=False,
        pretrained_model_path='', 
    )

    prev_trainable_params = count_trainable_params(net)
    net.freeze(freeze, 1)
    curr_trainable_params = count_trainable_params(net)
    if freeze:
        assert prev_trainable_params > curr_trainable_params
    else:
        assert prev_trainable_params == curr_trainable_params
