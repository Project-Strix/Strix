from configparser import ConfigParser
from pathlib import Path

from medlp.configures import config as cfg
from monai_ex.engines.utils import CustomKeys
from monai_ex.engines.utils import get_keys_dict as keys_dict

conf = ConfigParser()
conf.optionxform = str
fname = cfg.DEFAULT_MEDLP_CONFIG['CONFIG_FNAME']
cfg_file = Path(__file__).parent.joinpath(fname)


if not cfg_file.is_file():
    with cfg_file.open('w') as cfgfile:

        conf.add_section("MEDLP_CONFIG")
        for k, v in cfg.DEFAULT_MEDLP_CONFIG.items():
            conf.set('MEDLP_CONFIG', k.upper(), v)

        conf.add_section('CUSTOM_KEYS')
        for k, v in keys_dict(CustomKeys).items():
            conf.set('CUSTOM_KEYS', k.upper(), v)

        conf.add_section("GUI_CONFIG")

        conf.write(cfgfile)

cfg.init(add_path=True)
