from pathlib import Path
from configparser import ConfigParser


DEFAULT_MEDLP_CONFIG = {
    "MODE": 'dev',  # release
    "CONFIG_FNAME": 'medlp_configures.cfg',
    "OUTPUT_DIR": str(Path.home()/'MeDLP'/'medlp_exp'),
    "EXTERNAL_DATASET_DIR": str(Path.home()/'MeDLP'/'medlp_datasets'),
    "EXTERNAL_NETWORK_DIR": str(Path.home()/'MeDLP'/'medlp_networks'),
}


def __config_to_dict(config: ConfigParser):
    d = dict(config._sections)
    for k in d:
        d[k] = dict(d[k])
    return d


def init():
    global _config_dict

    try:
        fname = DEFAULT_MEDLP_CONFIG['CONFIG_FNAME']
        cfg_file = Path(__file__).parent.joinpath(fname)
        conf = ConfigParser()
        conf.optionxform = str

        conf.read(cfg_file)
        _config_dict = __config_to_dict(conf)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing config file {cfg_file}")
    except Exception as e:
        raise Exception(f"Error occured during reading config: {e}")


def get_cfg(section_name, keyword):
    return _config_dict[section_name][keyword.upper()]


def get_medlp_cfg(keyword):
    return _config_dict["MEDLP_CONFIG"][keyword.upper()]


def get_key(key_name):
    return _config_dict["CUSTOM_KEYS"][key_name.upper()]


def set_key(key_name, value):
    _config_dict["CUSTOM_KEYS"][key_name.upper()] = value


def get_keys_dict():
    return dict(_config_dict.get("CUSTOM_KEYS"))


def get_keys_list():
    return list(_config_dict.get("CUSTOM_KEYS").values())