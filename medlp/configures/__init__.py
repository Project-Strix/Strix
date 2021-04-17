from pathlib import Path
import configparser
from medlp.configures.config import DEFAULT_MEDLP_CONFIG

conf = configparser.ConfigParser()
fname = DEFAULT_MEDLP_CONFIG['config_fname']
cfg_file = Path(__file__).parent.joinpath(fname)

if not cfg_file.is_file():
    with cfg_file.open('w') as cfgfile:

        conf.add_section("MEDLP_CONFIG")
        for k, v in DEFAULT_MEDLP_CONFIG.items():
            conf.set('MEDLP_CONFIG', k, v)

        conf.add_section("GUI_CONFIG")

        conf.write(cfgfile)


def get_cfg(section_name, keyword):
    conf.read(cfg_file)
    return conf.get(section_name, keyword)
