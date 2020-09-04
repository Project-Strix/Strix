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

    from picc_detection import train
    from picc_detection import train_cfg
    main.add_command(train)
    main.add_command(train_cfg)
    main()