import os
import re


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
