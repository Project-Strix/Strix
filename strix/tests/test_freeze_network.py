import pytest

import torch
import torch.nn as nn

from strix.models import ARCHI_MAPPING
from strix.models.cnn.utils import set_trainable, count_trainable_params

@pytest.mark.parametrize("freeze", [True, False])
def test_segmentation_train_engine(freeze):
    model_type = ARCHI_MAPPING["multitask"]["3D"]["UnifiedNet-M3"]
    net = model_type(
        3,
        1,
        (1,1),
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
