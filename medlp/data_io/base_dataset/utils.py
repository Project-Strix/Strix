import os
from monai_ex.engines.utils import CustomKeys, get_keys_list


def get_input_data(files_list, is_supervised, verbose, dataset_name=''):
    """
    check input file_list format and existence.
    """
    if verbose:
        print('Custom keys:', get_keys_list(CustomKeys))

    input_data = []
    for f in files_list:
        if is_supervised:
            if isinstance(f, (list, tuple)):  # Recognize f as ['image','label']
                assert os.path.exists(f[0]), f"File not exists: {f[0]}"
                assert os.path.exists(f[1]), f"File not exists: {f[1]}"
                input_data.append({CustomKeys.IMAGE: f[0], CustomKeys.LABEL: f[1]})
            elif isinstance(f, dict):
                assert CustomKeys.IMAGE in f, f"File {f} doesn't contain image keyword '{CustomKeys.IMAGE}'"
                assert CustomKeys.LABEL in f, f"File {f} doesn't contain label keyword '{CustomKeys.LABEL}'"
                assert os.path.exists(f[CustomKeys.IMAGE]), f"File not exists: {f[CustomKeys.IMAGE]}"

                input_data.append(  # filter the dict by keys defined in CustomKeys
                    dict(filter(lambda k: k[0] in get_keys_list(CustomKeys), f.items()))
                )
            else:
                raise ValueError(
                    f"Not supported file_list format,"
                    f"Got {type(f)} in Supervised{dataset_name}"
                )
        else:
            if isinstance(f, str):
                assert os.path.exists(f), f"Image file not exists: {f}"
                input_data.append({CustomKeys.IMAGE: f})
            elif isinstance(f, dict):
                assert CustomKeys.IMAGE in f, f"File {f} doesn't contain image keyword '{CustomKeys.IMAGE}'"
                assert os.path.exists(f[CustomKeys.IMAGE]), f"File not exists: {f[CustomKeys.IMAGE]}"

                input_data.append(
                    dict(filter(lambda k: k[0] in get_keys_list(CustomKeys), f.items()))
                )
            else:
                raise ValueError(
                    f"Not supported file_list format,"
                    f"Got {type(f)} in Unsupervised{dataset_name}"
                )
    return input_data
