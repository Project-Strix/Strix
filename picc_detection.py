import os, sys, shutil, json, logging, warnings, time, random, torch
import numpy as np
from functools import partial
from torch import from_numpy, reshape, cuda, cat
from types import SimpleNamespace as sn
import nibabel as nib

from models import get_engine, get_test_engine
from data_io.dataio import get_dataloader, get_picc_datalist

from sklearn.model_selection import train_test_split
from utils_cw import Print, print_smi, confirmation, check_dir, recursive_glob2, prompt_when, get_items_from_file

import click
from click.parser import OptionParser
import click_callbacks as clb
from ignite.engine import Events

@click.command('train', context_settings={'allow_extra_args':True})
@click.option('--config', type=str, help="tmp var for train_from_cfg")
@click.option('--debug', is_flag=True)
@clb.latent_auxilary_params
@clb.common_params
@clb.network_params
@click.option('--transpose', type=int, nargs=2, default=None, help='Transpose data when loading')
@click.option('--smi', default=True, callback=print_smi, help='Print GPU usage')
@click.option('--gpus', prompt='Choose GPUs[eg: 0]', type=str, help='The ID of active GPU')
@click.option('--experiment-name', type=str, callback=clb.get_exp_name, default='')
@click.option('--confirm', callback=partial(confirmation, output_dir_ctx='experiment_name',save_code=True,exist_ok=False))
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
    files_train, files_valid = train_test_split(files_list, test_size=cargs.split, random_state=cargs.seed)
    Print(f'Get {len(files_train)} training data, {len(files_valid)} validation data', color='g')

    # Save param and datalist
    with open(os.path.join(cargs.out_dir, 'train_files'), 'w') as f:
        json.dump(files_train, f, indent=2)
    with open(os.path.join(cargs.out_dir, 'test_files'), 'w') as f:
        json.dump(files_valid, f, indent=2)

    train_loader = get_dataloader(cargs, files_train, phase='train')
    valid_loader = get_dataloader(cargs, files_valid, phase='valid')

    trainer = get_engine(cargs, train_loader, valid_loader, show_network=True)
    trainer.add_event_handler(event_name=Events.EPOCH_STARTED, handler=lambda x: print('-'*40))
    trainer.run()
        
@click.command('train-from-cfg', context_settings={'allow_extra_args':True, 'ignore_unknown_options':True})
@click.option('--config', type=click.Path(exists=True), help='Config file to load')
@click.argument('additional_args', nargs=-1, type=click.UNPROCESSED)
def train_cfg(**args):
    if len(args.get('additional_args')) != 0: #parse additional args
        Print('*** Lr schedule changes do not work yet! Please make a confirmation at last!***\n', color='y')

    configures = get_items_from_file(args['config'], format='json')
    #click.confirm(f"Loading configures: {configures}", default=True, abort=True, show_default=True)
    
    #Convert args to index
    configures['data_list'] = clb.dataset_list.index(configures['data_list'])
    configures['model_type'] = clb.model_types.index(configures['model_type'])
    configures['criterion'] = clb.losses.index(configures['criterion'])
    configures['lr_policy'] = clb.lr_schedule.index(configures['lr_policy'])
    configures['framework'] = clb.framework_types.index(configures['framework'])
    #configures['layer_order'] = clb.layer_orders.index(configures['layer_order'])
    configures['smi'] = False
    gpu_id = click.prompt(f"Current GPU id: {configures['gpus']}")
    configures['gpus'] = gpu_id
    
    train(default_map=configures)
    #ctx.invoke(train, **configures) 


@click.command('test-from-cfg')
@click.option('--config', type=click.Path(exists=True), help='Config file to load')
@click.option('--smi', default=True, callback=print_smi, help='Print GPU usage')
@click.option('--gpus', prompt='Choose GPUs[eg: 0]', type=str, help='The ID of active GPU')
def test_cfg(**args):
    configures = get_items_from_file(args['config'], format='json')
    
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        Print('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpus'])

    exp_dir = args.get('experiment_name', os.path.dirname(args['config']))
    assert os.path.isfile(os.path.join(exp_dir, 'test_files')), f'Test file does not exists in {exp_dir}!'
    test_files = get_items_from_file(os.path.join(exp_dir, 'test_files'), format='json')
    configures['model_path'] = clb.get_trained_models(exp_dir)
    configures['out_dir'] = check_dir(exp_dir, 'Test')
    configures['preload'] = False
    
    test_loader = get_dataloader(sn(**configures), test_files, phase='test')

    engine = get_test_engine(sn(**configures), test_loader)
    Print("Begin testing...", color='g')
    engine.run()