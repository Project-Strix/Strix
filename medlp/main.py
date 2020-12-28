import sys, click

import warnings
warnings.filterwarnings("ignore")


@click.group()
def main():
    '''Main entry of this program.

    For training:
        python3 picc.py train --params

    For training from config file:
        python3 picc.py train_from_cfg --params
    '''
    # model = __init__(args)


if __name__ == '__main__':
    assert sys.version_info >= (3, 6), "Python ver. >=3.6 is required!"

    from medlp.main_entry import train
    from medlp.main_entry import train_cfg
    from medlp.main_entry import test_cfg
    from medlp.main_entry import unlink_dirs
    from nni_search import nni_search
    from nni_search import train_nni
    main.add_command(train)
    main.add_command(train_cfg)
    main.add_command(test_cfg)
    main.add_command(unlink_dirs)
    main.add_command(nni_search)
    main.add_command(train_nni)
    main()