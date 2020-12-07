import tqdm, math, random, time
from PIL import Image
import inspect, re, os, h5py, collections, json, csv
import numpy as np
from skimage.exposure import rescale_intensity
from utils_cw import Print, load_h5, check_dir

from medlp.data_io import DATASET_MAPPING
from monai.data import DataLoader
from monai.transforms import (
        Compose,
        LoadHdf5d,
        LoadNumpyd,
        AddChanneld,
        RandCropByPosNegLabeld,
        RepeatChanneld,
        Lambdad,
        ToTensord
    )

def get_datalist(dataset_name):
    if dataset_name == 'picc_h5':
        if os.name == 'nt':
            fname = r"\\mega\clwang\Data\picc\prepared_h5\data_list.json"
        elif os.name == 'posix':
            fname = "/homes/clwang/Data/picc/prepared_h5/data_list_linux.json"
    elif dataset_name == 'jsph_rcc':
        fname = "/homes/clwang/Data/jsph_rcc/kidney_rcc/Train/data_list.json"
    else:
        raise ValueError
    
    return fname


def get_default_setting(phase, **kwargs):
    if phase == 'train': #Todo: move this part to each dataset
        shuffle = kwargs.get('train_shuffle', True)
        batch_size = kwargs.get('train_n_batch', 5)
        num_workers = kwargs.get('train_n_workers', 10)
        drop_last = kwargs.get('train_drop_last', True)
        pin_memory = kwargs.get('train_pin_memory', True)
    elif phase == 'valid':
        shuffle =  kwargs.get('valid_shuffle', True)
        batch_size = kwargs.get('valid_n_batch', 2)
        num_workers = kwargs.get('valid_n_workers', 2)
        drop_last = kwargs.get('valid_drop_last', False)
        pin_memory = kwargs.get('valid_pin_memory', True)
    elif phase == 'test' or phase == 'test_wo_label':
        shuffle = kwargs.get('test_shuffle', False)
        batch_size = kwargs.get('test_n_batch', 1)
        num_workers = kwargs.get('test_n_workers', 2)
        drop_last = kwargs.get('test_drop_last', False)
        pin_memory = kwargs.get('test_pin_memory', True)
    else:
        raise ValueError(f"phase must be in 'train,valid,test', but got {phase}") 
    
    return {'batch_size':batch_size, 'shuffle':shuffle, 'drop_last':drop_last, 'num_workers':num_workers, 'pin_memory':pin_memory}


def get_dataloader(args, files_list, phase='train'):
    params = get_default_setting(phase, train_n_batch=args.n_batch, valid_n_batch=1) #! How to customize?
    arguments = {
        'files_list':files_list,
        'phase': phase,
        'opts': vars(args)
    }

    dataset_ = DATASET_MAPPING[args.framework][args.tensor_dim][args.data_list](**arguments)
    
    loader = DataLoader(dataset_, **params)
    return loader

