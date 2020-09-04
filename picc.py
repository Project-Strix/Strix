import sys, click

import warnings
warnings.filterwarnings("ignore")


@click.group()
def main():
    '''Main entry of this program.

    For training:
        python3 main.py train --params
    For prediction:
        python3 main.py predict --params
    '''
    # model = __init__(args)


if __name__ == '__main__':
    assert sys.version_info >= (3, 6), "Python ver. >=3.6 is required!"

    from picc_detection import train
    #from picc_detection import predict
    main.add_command(train)
    #main.add_command(predict)
    main(obj={})