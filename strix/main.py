import sys
import click

import warnings

warnings.filterwarnings("ignore")


@click.group()
def main():
    """Main entry of this program.

    For training:
        python3 picc.py train --params

    For training from config file:
        python3 picc.py train_from_cfg --params
    """
    # model = __init__(args)


if __name__ == "__main__":
    assert sys.version_info >= (3, 8), "Python ver. >=3.8 is required!"

    from strix.main_entry import train
    from strix.main_entry import train_cfg
    from strix.main_entry import test_cfg
    from strix.main_entry import train_and_test
    from nni_search import nni_search
    from nni_search import train_nni
    from data_checker import check_data
    from interpreter import gradcam
    from tools import merge_roc_curves, summarize_data

    main.add_command(train)
    main.add_command(train_cfg)
    main.add_command(test_cfg)
    main.add_command(train_and_test)
    main.add_command(nni_search)
    main.add_command(train_nni)
    main.add_command(check_data)
    main.add_command(gradcam)
    main.add_command(merge_roc_curves)
    main.add_command(summarize_data)
    main()
