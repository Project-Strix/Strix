import os
# from monai_ex.engines.utils import CustomKeys
# from medlp.configures import get_keys_dict, get_keys_list, cfg.get_key
from medlp.configures import config as cfg

def get_input_data(files_list, is_supervised, verbose, dataset_name=''):
    """
    check input file_list format and existence.
    """
    if verbose:
        print('Custom keys:', cfg.get_keys_dict())

    input_data = []
    for f in files_list:
        if is_supervised:
            if isinstance(f, (list, tuple)):  # Recognize f as ['image','label']
                assert os.path.exists(f[0]), f"File not exists: {f[0]}"
                assert os.path.exists(f[1]), f"File not exists: {f[1]}"
                input_data.append({cfg.get_key("IMAGE"): f[0], cfg.get_key("LABEL"): f[1]})
            elif isinstance(f, dict):
                assert cfg.get_key("IMAGE") in f, f"File {f} doesn't contain image keyword '{cfg.get_key('IMAGE')}'"
                assert cfg.get_key("LABEL") in f, f"File {f} doesn't contain label keyword '{cfg.get_key('LABEL')}'"
                assert os.path.exists(f[cfg.get_key("IMAGE")]), f"File not exists: {f[cfg.get_key('IMAGE')]}"

                input_data.append(  # filter the dict by keys defined in CustomKeys
                    dict(filter(lambda k: k[0] in cfg.get_keys_list(), f.items()))
                )

            else:
                raise ValueError(
                    f"Not supported file_list format,"
                    f"Got {type(f)} in Supervised{dataset_name}"
                )
        else:
            if isinstance(f, str):
                assert os.path.exists(f), f"Image file not exists: {f}"
                input_data.append({cfg.get_key('IMAGE'): f})
            elif isinstance(f, dict):
                assert cfg.get_key('IMAGE') in f, f"File {f} doesn't contain image keyword '{cfg.get_key('IMAGE')}'"
                assert os.path.exists(f[cfg.get_key('IMAGE')]), f"File not exists: {f[cfg.get_key('IMAGE')]}"

                input_data.append(
                    dict(filter(lambda k: k[0] in cfg.get_keys_list(), f.items()))
                )
            else:
                raise ValueError(
                    f"Not supported file_list format,"
                    f"Got {type(f)} in Unsupervised{dataset_name}"
                )
    return input_data
