import re
from monai.networks import one_hot
import torch
from monai_ex.handlers import from_engine_ex as from_engine

def get_best_model(folder, float_regex=r"=(-?\d+\.\d+).pt"):
    models = list(
        filter(
            lambda x: x.is_file(),
            [model for model in (folder / "Models" / "Best_Models").iterdir()],
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
        if model_type == "best":
            best_models.append(get_best_model(folder))
        elif model_type == "last":
            best_models.append(get_last_model(folder))
        else:
            raise ValueError(
                "Only support 'best' and 'last' model types,"
                f"but got '{model_type}' type"
            )
    return best_models


# Todo: refactor this function
def output_onehot_transform(output, n_classes=3, verbose=False):
    y_pred, y = output["pred"], output["label"]
    if verbose:
        print(
            "Input y_pred:",
            list(y_pred.cpu().numpy()),
            "\nInput y_ture:",
            list(y.cpu().numpy()),
        )

    if n_classes == 1:
        return y_pred, y

    def onehot_(data, n_class):
        if data.ndimension() == 1:
            data_ = one_hot(data, n_class)
        elif data.ndimension() == 2:  # first dim is batch
            data_ = one_hot(data, n_class, dim=1)
        elif data.ndimension() == 3 and data.shape[1] == 1:
            data_ = one_hot(data.squeeze(1), n_class, dim=1)
        else:
            raise ValueError(
                f"Cannot handle data ndim: {data.ndimension()}, shape: {data.shape}"
            )
        return data_

    pred_ = onehot_(y_pred, n_classes)
    true_ = onehot_(y, n_classes)

    assert (
        pred_.shape == true_.shape
    ), f"Pred ({pred_.shape}) and True ({true_.shape}) data have different shape"
    return pred_, true_


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


def get_dice_metric_transform_fn(output_nc, pred_key, label_key, decollate):
    if output_nc > 1:
        ch_dim = 0 if decollate else 1
        key_metric_transform_fn = from_engine(
            [pred_key, label_key],
            transforms=[
                lambda x: x,
                lambda x: one_hot(x, num_classes=output_nc, dim=ch_dim)
            ]
        )
    else:
        key_metric_transform_fn = from_engine([pred_key, label_key])

    return key_metric_transform_fn
