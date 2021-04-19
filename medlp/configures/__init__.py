import configparser
from pathlib import Path

from medlp.configures.config import DEFAULT_MEDLP_CONFIG
from monai_ex.engines.utils import CustomKeys, get_keys_dict

conf = configparser.ConfigParser()
conf.optionxform = str
fname = DEFAULT_MEDLP_CONFIG['CONFIG_FNAME']
cfg_file = Path(__file__).parent.joinpath(fname)

if not cfg_file.is_file():
    with cfg_file.open('w') as cfgfile:

        conf.add_section("MEDLP_CONFIG")
        for k, v in DEFAULT_MEDLP_CONFIG.items():
            conf.set('MEDLP_CONFIG', k, v)

        conf.add_section('CUSTOM_KEYS')
        for k, v in get_keys_dict(CustomKeys).items():
            conf.set('CUSTOM_KEYS', k, v)

        conf.add_section("GUI_CONFIG")

        conf.write(cfgfile)


def get_cfg(section_name, keyword):
    assert cfg_file.is_file(), f"{fname} file not exists!"
    conf.read(cfg_file)
    return conf.get(section_name, keyword)


def get_key(key_name):
    assert cfg_file.is_file(), f"{fname} file not exists!"
    conf.read(cfg_file)
    return conf.get("CUSTOM_KEYS", key_name.upper())
