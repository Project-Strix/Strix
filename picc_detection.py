import os, sys, functools, shutil, json, logging
import numpy as np
import torch
from torch import from_numpy, reshape, cuda, cat
from types import SimpleNamespace as sn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utilities.dataio import get_dataloader, get_picc_datalist
from sklearn.model_selection import train_test_split
from utils_cw import Print, print_smi, confirmation, check_dir, recursive_glob2, prompt_when, get_items_from_file
import nibabel as nib

from models import get_engine
import click
from click.parser import OptionParser
import click_callbacks as clb

@click.command('train', context_settings={'allow_extra_args':True})
@click.option('--config', type=str, help="tmp var for train_from_cfg")
@clb.common_params
@clb.network_params
@click.option('--debug', is_flag=True)
@click.option('--snip', is_flag=True)
@click.option('--snip_percent', type=float, default=0.4, callback=functools.partial(prompt_when,trigger='snip'), help='Pruning ratio of wights/channels')
@click.option('--preload', type=bool, default=True, help='Preload all data once')
@click.option('--transpose', type=int, nargs=2, default=None, help='Transpose data when loading')
@click.option('-p', '--partial', type=float, default=1, callback=functools.partial(prompt_when,trigger='debug'), help='Only load part of data')
@click.option('--save-epoch-freq', type=int, default=20, help='Save model freq')
@click.option('--seed', type=int, default=100, help='random seed')
@click.option('--smi', default=True, callback=print_smi, help='Print GPU usage')
@click.option('--gpus', prompt='Choose GPUs[eg: 0]', type=str, help='The ID of active GPU')
@click.option('--experiment-name', type=str, callback=clb.get_exp_name, default='')
@click.option('--confirm', callback=functools.partial(confirmation, output_dir_ctx='experiment_name',save_code=True))
def train(**args):
    cargs = sn(**args)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        Print('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cargs.gpus)

    cargs.gpu_ids = list(range(len(list(map(int,cargs.gpus.split(','))))))
    cargs.out_dir = cargs.experiment_name

    data_list = get_picc_datalist(cargs.data_list)
    assert os.path.isfile(data_list), 'Data list not exists!'
    files_list = get_items_from_file(data_list, format='json')
    if cargs.partial < 1:
        Print('Use {} data'.format(int(len(files_list)*cargs.partial)), color='y')
        files_list = files_list[:int(len(files_list)*cargs.partial)]
    cargs.split = int(cargs.split) if cargs.split > 1 else cargs.split
    files_train, files_test = train_test_split(files_list, test_size=cargs.split, random_state=cargs.seed)

    # Save param and datalist
    with open(os.path.join(cargs.out_dir, 'train_files'), 'w') as f:
        json.dump(files_train, f, indent=2)
    with open(os.path.join(cargs.out_dir, 'test_files'), 'w') as f:
        json.dump(files_test, f, indent=2)

    train_loader = get_dataloader(cargs, files_train, dataset_type='train')
    test_loader  = get_dataloader(cargs, files_test, dataset_type='valid')

    engine = get_engine(cargs, train_loader, test_loader, show_network=True)
    engine.run()
        
@click.command('train_from_cfg', context_settings={'allow_extra_args':True, 'ignore_unknown_options':True})
@click.option('--config', type=click.Path(exists=True), help='Config file to load')
@click.argument('additional_args', nargs=-1, type=click.UNPROCESSED)
def train_cfg(**args):
    #if len(args.get('additional_args')) != 0: #parse additional args
        # for i in range(0, len(args['additional_args']), 2):
        #     print(args['additional_args'][i].strip('-').replace('-','_'))

    configures = get_items_from_file(args['config'], format='json')
    #click.confirm(f"Loading configures: {configures}", default=True, abort=True, show_default=True)
    configures['data_list'] = clb.dataset_list.index(configures['data_list'])
    configures['model_type'] = clb.model_types.index(configures['model_type'])
    configures['criterion'] = clb.losses.index(configures['criterion'])
    configures['lr_policy'] = clb.lr_schedule.index(configures['lr_policy'])
    configures['framework'] = clb.framework_types.index(configures['framework'])
    configures['layer_order'] = clb.layer_orders.index(configures['layer_order'])
    
    train(default_map=configures)
    #ctx.invoke(train, **configures) 

