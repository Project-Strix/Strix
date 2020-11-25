import os, sys, shutil, json, logging, warnings, time, random, torch
import numpy as np
from functools import partial
from torch import from_numpy, reshape, cuda, cat
from torch.utils.tensorboard import SummaryWriter
from types import SimpleNamespace as sn
import nibabel as nib

from medlp.models import get_engine, get_test_engine
from medlp.data_io.dataio import get_dataloader, get_datalist
from medlp.utilities.handlers import TensorboardGraph, SNIP_prune_handler
import medlp.utilities.click_callbacks as clb
from medlp.utilities import enum

from sklearn.model_selection import train_test_split, KFold
from utils_cw import Print, print_smi, confirmation, check_dir, recursive_glob2, prompt_when, get_items_from_file

import click
from ignite.engine import Events
from ignite.utils import setup_logger
from monai.handlers import CheckpointLoader

@click.command('train', context_settings={'allow_extra_args':True})
@click.option('--config', type=str, help="tmp var for train_from_cfg")
@click.option('--debug', is_flag=True)
@clb.latent_auxilary_params
@clb.common_params
@clb.solver_params
@clb.network_params
#@click.option('--transpose', type=int, nargs=2, default=None, help='Transpose data when loading')
@click.option('--smi', default=True, callback=print_smi, help='Print GPU usage')
@click.option('--gpus', prompt='Choose GPUs[eg: 0]', type=str, help='The ID of active GPU')
@click.option('--experiment-path', type=str, callback=clb.get_exp_name, default='')
@click.option('--confirm', callback=partial(confirmation, output_dir_ctx='experiment_path',save_code=True,exist_ok=False))
def train(**args):
    cargs = sn(**args)
    
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        Print('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cargs.gpus)

    cargs.gpu_ids = list(range(len(list(map(int,cargs.gpus.split(','))))))

    data_list = get_datalist(cargs.data_list)
    assert os.path.isfile(data_list), 'Data list not exists!'
    files_list = get_items_from_file(data_list, format='json')
    if cargs.partial < 1:
        Print('Use {} data'.format(int(len(files_list)*cargs.partial)), color='y')
        files_list = files_list[:int(len(files_list)*cargs.partial)]
    
    if cargs.n_fold > 0:
        Print(f'Processing {cargs.n_fold} cross-validation', color='g')
        kf = KFold(n_splits=cargs.n_fold, random_state=cargs.seed, shuffle=True)
        all_folds = [ (train_index, test_index) for train_index, test_index in kf.split(files_list)]
        files_train = list(np.array(files_list)[all_folds[cargs.ith_fold][0]])
        files_valid = list(np.array(files_list)[all_folds[cargs.ith_fold][1]])
    else:
        cargs.split = int(cargs.split) if cargs.split > 1 else cargs.split
        files_train, files_valid = train_test_split(files_list, test_size=cargs.split, random_state=cargs.seed)
    Print(f'Get {len(files_train)} training data, {len(files_valid)} validation data', color='g')

    # Save param and datalist
    with open(os.path.join(cargs.experiment_path, 'train_files'), 'w') as f:
        json.dump(files_train, f, indent=2)
    with open(os.path.join(cargs.experiment_path, 'test_files'), 'w') as f:
        json.dump(files_valid, f, indent=2)

    train_loader = get_dataloader(cargs, files_train, phase='train')
    valid_loader = get_dataloader(cargs, files_valid, phase='valid')

    # Tensorboard Logger
    writer = SummaryWriter(log_dir=os.path.join(cargs.experiment_path, 'tensorboard'))
    if not cargs.debug:
        tb_dir = check_dir(os.path.dirname(cargs.experiment_path),'tb')
        target_dir = os.path.join(tb_dir, os.path.basename(cargs.experiment_path))
        if os.path.islink(target_dir):
            os.unlink(target_dir)

        os.symlink(os.path.join(cargs.experiment_path, 'tensorboard'), target_dir, target_is_directory=True)

    trainer, net, loss_fn = get_engine(cargs, train_loader, valid_loader, writer=writer, show_network=cargs.visualize)

    logging_level = logging.DEBUG if cargs.debug else logging.INFO
    trainer.logger = setup_logger(f'{cargs.tensor_dim}-Trainer', level=logging_level)
    if not cargs.verbose_log and not cargs.debug:
        logging.StreamHandler.terminator = "\r"
    
    trainer.add_event_handler(event_name=Events.EPOCH_STARTED, handler=lambda x: print('\n','-'*40))

    if os.path.isfile(cargs.pretrained_model_path):
        Print("Load pretrained model for contiune training:\n\t", cargs.pretrained_model_path, color='g')
        trainer.add_event_handler(event_name=Events.STARTED, 
                                  handler=CheckpointLoader(load_path=cargs.pretrained_model_path,
                                                           load_dict={"net": net}, strict=False, skip_mismatch=True))
    if cargs.visualize:
        Print('Visualize the architecture to tensorboard', color='g')
        trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED(once=1),
                                  handler=TensorboardGraph(net, writer, lambda x:x['image']))

    if cargs.snip:
        Print('Begin SNIP pruning', color='g')
        device = torch.device("cuda") if cargs.gpus != '-1' else torch.device("cpu")
        trainer.add_event_handler(event_name=Events.ITERATION_STARTED(once=1),
                                handler=SNIP_prune_handler(net, loss_fn, 0.3, train_loader, lambda x:x['image'], device=device))

    trainer.run()


@click.command('train-from-cfg', context_settings={'allow_extra_args':True, 'ignore_unknown_options':True})
@click.option('--config', type=click.Path(exists=True), help='Config file to load')
@click.option('--n-fold', type=int, default=0, help='K fold cross-validation')
@click.argument('additional_args', nargs=-1, type=click.UNPROCESSED)
def train_cfg(**args):
    if len(args.get('additional_args')) != 0: #parse additional args
        Print('*** Lr schedule changes do not work yet! Please make a confirmation at last!***\n', color='y')

    configures = get_items_from_file(args['config'], format='json')
    #click.confirm(f"Loading configures: {configures}", default=True, abort=True, show_default=True)
    
    configures['smi'] = False
    gpu_id = click.prompt(f"Current GPU id: {configures['gpus']}")
    configures['gpus'] = gpu_id
    
    if args['n_fold'] > 0:
        for i in range(args['n_fold']):
            configures['ith_fold'] = i
            train(default_map=configures)
    else:
        train(default_map=configures)
        #ctx.invoke(train, **configures) 


@click.command('test-from-cfg')
@click.option('--config', type=click.Path(exists=True), help='Config file to load')
@click.option('--test-files', type=str, default='', help='External files (.json) for testing')
@click.option('--with-label', is_flag=True, help='whether test data has label')
@click.option('--save-image', is_flag=True, help='Save the tested image data')
@click.option('--smi', default=True, callback=print_smi, help='Print GPU usage')
@click.option('--gpus', prompt='Choose GPUs[eg: 0]', type=str, help='The ID of active GPU')
def test_cfg(**args):
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    
    configures = get_items_from_file(args['config'], format='json')
    
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        Print('CUDA_VISIBLE_DEVICES specified, ignoring --gpu flag')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args['gpus'])

    exp_dir = configures.get('experiment_path', os.path.dirname(args['config']))
    if os.path.isfile(args['test_files']):
        test_fpath = args['test_files']
        test_files = get_items_from_file(args['test_files'], format='json')
    else:
        assert os.path.isfile(os.path.join(exp_dir, 'test_files')), f'Test file does not exists in {exp_dir}!'
        test_fpath = os.path.join(exp_dir, 'test_files')
        test_files = get_items_from_file(os.path.join(exp_dir, 'test_files'), format='json')

    configures['model_path'] = clb.get_trained_models(exp_dir)
    configures['experiment_path'] = check_dir(exp_dir, f'Test@{time.strftime("%m%d_%H%M")}')
    configures['preload'] = 0.0
    phase = 'test' if args['with_label'] else 'test_wo_label'
    configures['phase'] = phase
    configures['save_image'] = args['save_image']
    
    Print(f'{len(test_files)} test files', color='g')
    test_loader = get_dataloader(sn(**configures), test_files, phase=phase)

    engine = get_test_engine(sn(**configures), test_loader)
    engine.logger = setup_logger(f"{configures['tensor_dim']}-Tester", level=logging.INFO)

    Print("Begin testing...", color='g')
    shutil.copyfile(test_fpath, os.path.join(configures['experiment_path'], os.path.basename(test_fpath)))
    engine.run()


@click.command('unlink')
@click.option('--root-dir', type=click.Path(exists=True), help='Root dir contains symbolic dirs')
@click.option('-a', '--all-dir', is_flag=True, help='Unlink all dirs including both avalible and unavailable dirs')
def unlink_dirs(root_dir, all_dir):
    for d in os.listdir(root_dir):
        d = os.path.join(root_dir, d)
        if os.path.islink(d):
            if not os.path.isdir(d):
                os.unlink(d)
                print('Unlinked unavailable symbolic dir:', d)
            elif all_dir:
                os.unlink(d)
                print('Unlinked symbolic dir:', d)
            else:
                pass
        
