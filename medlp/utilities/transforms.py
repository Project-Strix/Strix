from typing import Sequence, Callable
from monai.transforms import apply_transform

def decollate_transform_adaptor(transfrom_fn: Callable):
    """Adaptor for transforms to handle decollated data.
    Making sure input data is a decollated sequence type.

    Args:
        transfrom_fn (Callable): target transform fn to be adapted.
    """
    def _inner(input_data: Sequence):
        if isinstance(input_data, Sequence):
            return [apply_transform(transfrom_fn, data) for data in input_data]
        else:
            # if isinstance(input_data, dict) and 'pred' in input_data:
            #     print(type(input_data.get('pred')), input_data.get('pred'))
            return apply_transform(transfrom_fn, input_data)
    return _inner
