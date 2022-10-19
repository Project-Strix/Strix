import sys
from pathlib import Path
from configparser import ConfigParser


DEFAULT_STRIX_CONFIG = {
    "MODE": "dev",  # release
    "CONFIG_FNAME": "strix_configures.cfg",
    "CACHE_DIR": str(Path.home() / "Strix"),
    "OUTPUT_DIR": str(Path.home() / "Strix" / "strix_exp"),
    "EXTERNAL_DATASET_DIR": str(Path.home() / "Strix-Contrib" / "Datasets"),
    "EXTERNAL_NETWORK_DIR": str(Path.home() / "Strix-Contrib" / "Networks"),
    "EXTERNAL_LOSS_DIR": str(Path.home() / "Strix-Contrib" / "Losses"),
}


def __config_to_dict(config: ConfigParser):
    d = dict(config._sections)
    for k in d:
        d[k] = dict(d[k])
    return d


def init(add_path=True):
    global _config_dict

    try:
        fname = DEFAULT_STRIX_CONFIG["CONFIG_FNAME"]
        cfg_file = Path(__file__).parent.joinpath(fname)
        conf = ConfigParser()
        conf.optionxform = str

        conf.read(cfg_file)
        _config_dict = __config_to_dict(conf)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing config file {cfg_file}")
    except Exception as e:
        raise Exception(f"Error occured during reading config: {e}")

    if add_path:
        sys.path.append(_config_dict["STRIX_CONFIG"]["EXTERNAL_DATASET_DIR"])
        sys.path.append(_config_dict["STRIX_CONFIG"]["EXTERNAL_NETWORK_DIR"])
        sys.path.append(_config_dict["STRIX_CONFIG"]["EXTERNAL_LOSS_DIR"])


def get_cfg(section_name, keyword):
    return _config_dict[section_name][keyword.upper()]


def get_strix_cfg(keyword):
    return _config_dict["STRIX_CONFIG"][keyword.upper()]


def get_key(key_name):
    return _config_dict["CUSTOM_KEYS"][key_name.upper()]


def set_key(key_name, value):
    _config_dict["CUSTOM_KEYS"][key_name.upper()] = value


def get_keys_dict():
    return dict(_config_dict.get("CUSTOM_KEYS"))


def get_keys_list():
    return list(_config_dict.get("CUSTOM_KEYS").values())