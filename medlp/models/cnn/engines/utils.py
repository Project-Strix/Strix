import re
import torch


def get_best_model(folder, float_regex=r"=(-?\d+\.\d+).pt"):
    models = list(
        filter(
            lambda x: x.is_file(),
            [model for model in folder.joinpath("Models").iterdir()],
        )
    )
    if len(models) == 0:
        return None
    else:
        models.sort(key=lambda x: float(re.search(float_regex, x.name).group(1)))
        return models[-1]


def get_last_model(folder, int_regex = r"=(\d+).pt"):
    models = list(
        filter(
            lambda x: x.is_file(),
            [model for model in (folder / "Models" / "Checkpoint").iterdir()],
        )
    )
    try:
        models.sort(key=lambda x: int(re.search(int_regex, x.name).group(1)))
    except AttributeError as e:
        invalid_models = list(
            filter(lambda x: re.search(int_regex, x.name) is None, models)
        )
        print("invalid models:", invalid_models)
        raise e

    return models[-1]


def get_models(folders, model_type):
    best_models = []
    for folder in folders:
        if model_type == 'best':
            best_models.append(get_best_model(folder))
        elif model_type == 'last':
            best_models.append(get_last_model(folder))
        else:
            raise ValueError(
                "Only support 'best' and 'last' model types,"
                f"but got '{model_type}' type"
            )
    return best_models


def get_prepare_batch_fn(
    opts, image_key, label_key, multi_input_keys, multi_output_keys
): 
    target_type = torch.FloatTensor
    if opts.criterion in ["BCE", "WBCE", "FocalLoss"]:
        target_type = torch.FloatTensor
    elif opts.criterion in ["CE", "WCE"]:
        target_type = torch.LongTensor

    if multi_input_keys is not None and multi_output_keys is not None:
        prepare_batch_fn = lambda x, device, nb: (
            tuple(x[key].to(device) for key in multi_input_keys),
            tuple(x[key].type(target_type).to(device) for key in multi_output_keys),
        )
    elif multi_input_keys is not None:
        prepare_batch_fn = lambda x, device, nb: (
            tuple(x[key].to(device) for key in multi_input_keys),
            x[label_key].type(target_type).to(device),
        )
    elif multi_output_keys is not None:
        prepare_batch_fn = lambda x, device, nb: (
            x[image_key].to(device),
            tuple(x[key].type(target_type).to(device) for key in multi_output_keys),
        )
    else:
        prepare_batch_fn = lambda x, device, nb: (
            x[image_key].to(device),
            x[label_key].type(target_type).to(device),
        )

    return prepare_batch_fn


def get_unsupervised_prepare_batch_fn(opts, image_key, multi_input_keys):
    if multi_input_keys is not None:
        prepare_batch_fn = lambda x, device, nb: (
            tuple(x[key].to(device) for key in multi_input_keys),
            None,
        )
    else:
        prepare_batch_fn = lambda x, device, nb: (x[image_key].to(device), None)

    return prepare_batch_fn
