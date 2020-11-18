import tqdm, math, random, time
from PIL import Image
import inspect, re, os, h5py, collections, json, csv
import numpy as np
from skimage.exposure import rescale_intensity
from utils_cw import Print, load_h5, check_dir

from medlp.data_io.picc_dataset import *
from medlp.data_io.dr_sl_dataset import get_ObjCXR_dataset, get_NIHXray_dataset
from medlp.data_io.kits_dataset import get_kits_dataset
from medlp.data_io.rjh_dataset import get_rjh_tswi_seg_dataset


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
    elif dataset_name == 'picc_nii':
        fname = "/homes/clwang/Data/picc/picc_seg_nii.json"
    elif dataset_name == 'picc_dcm':
        fname = "/homes/clwang/Data/picc/picc_dcm_nii.json"
    elif dataset_name == 'Obj_CXR':
        fname = "/homes/clwang/Data/object-CXR/train_data_list.json"
    elif dataset_name == 'NIH_CXR':
        fname = "/homes/clwang/Data/NIH-CXR_TRAIN_VAL_LIST.json"
    elif dataset_name == 'rib':
        fname = "/homes/clwang/Data/picc/prepared_rib_h5/nii_files.json"
    elif dataset_name == 'kits':
        fname = '/MRIData/kits19/data/train_data_list.json'
    elif dataset_name == 'jsph_rcc':
        fname = "/homes/clwang/Data/jsph_rcc/kidney_rcc/Train/data_list.json"
    elif dataset_name == 'rjh_tswi':
        fname = "/homes/clwang/Data/RJH/RJ_data/preprocessed/labeled_data_list.json"
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
    if args.data_list == 'rib':
        params = get_default_setting(phase, train_n_batch=args.n_batch, valid_n_batch=1)
        dataset_ = RIB_seg_dataset(files_list,
                                   phase=phase,
                                   in_channels=args.input_nc,
                                   preload=args.preload,
                                   image_size=args.image_size,
                                   crop_size=args.crop_size,
                                   augment_ratio=args.augment_ratio,
                                   downsample=args.downsample,
                                   verbose=args.debug
                                   )
    elif args.data_list == 'picc_h5':
        params = get_default_setting(phase, train_n_batch=args.n_batch)
        dataset_ = PICC_seg_dataset(files_list,
                                    phase=phase,
                                    spacing=args.spacing,
                                    in_channels=args.input_nc,
                                    image_size=args.image_size,
                                    crop_size=args.crop_size,
                                    preload=args.preload,
                                    augment_ratio=args.augment_ratio,
                                    downsample=args.downsample,
                                    verbose=args.debug
                                    )
    elif args.data_list == 'picc_nii':
        params = get_default_setting(phase, train_n_batch=args.n_batch)
        dataset_ = PICC_nii_seg_dataset(files_list,
                                        phase=phase,
                                        spacing=args.spacing,
                                        in_channels=args.input_nc,
                                        image_size=args.image_size,
                                        crop_size=args.crop_size,
                                        preload=args.preload,
                                        augment_ratio=args.augment_ratio
                                        )
    elif args.data_list == 'picc_dcm':
        params = get_default_setting(phase, train_n_batch=args.n_batch)
        dataset_ = PICC_dcm_seg_dataset(files_list,
                                        phase=phase,
                                        spacing=args.spacing,
                                        in_channels=args.input_nc,
                                        image_size=args.image_size,
                                        crop_size=args.crop_size,
                                        preload=args.preload,
                                        augment_ratio=args.augment_ratio
                                        )
    elif args.data_list == 'Obj_CXR':
        params = get_default_setting(phase, train_n_batch=args.n_batch, valid_n_batch=args.n_batch, valid_n_workers=10)
        dataset_ = get_ObjCXR_dataset(files_list, 
                                      phase=phase, 
                                      in_channels=args.input_nc, 
                                      preload=args.preload, 
                                      image_size=args.image_size,
                                      crop_size=args.crop_size, 
                                      augment_ratio=args.augment_ratio, 
                                      verbose=args.debug
                                      )
    elif args.data_list == 'NIH_CXR':
        params = get_default_setting(phase, train_n_batch=args.n_batch, valid_n_batch=args.n_batch, valid_n_workers=10)
        dataset_ = get_NIHXray_dataset(files_list,
                                       phase=phase,
                                       in_channels=args.input_nc,
                                       preload=args.preload,
                                       image_size=args.image_size,
                                       crop_size=args.crop_size,
                                       augment_ratio=args.augment_ratio,
                                       verbose=args.debug
                                       )
    elif args.data_list == 'kits':
        params = get_default_setting(phase, train_n_batch=args.n_batch, valid_n_batch=args.n_batch, valid_n_workers=5)
        dataset_  = get_kits_dataset(files_list, 
                                     phase=phase,
                                     spacing=args.spacing,
                                     winlevel=[-80,304],
                                     in_channels=args.input_nc,
                                     crop_size=args.crop_size,
                                     preload=args.preload,
                                     augment_ratio=args.augment_ratio,
                                     cache_dir=check_dir(args.experiment_path,'caches'),
                                     verbose=args.debug
                                     )
    elif args.data_list == 'jsph_rcc':
        params = get_default_setting(phase, train_n_batch=args.n_batch, valid_n_batch=args.n_batch, valid_n_workers=5)
        dataset_  = get_kits_dataset(files_list, 
                                     phase=phase,
                                     spacing=args.spacing,
                                     winlevel=[-80,304],
                                     in_channels=args.input_nc,
                                     crop_size=args.crop_size,
                                     preload=args.preload,
                                     augment_ratio=args.augment_ratio,
                                     cache_dir=check_dir(args.experiment_path,'caches'),
                                     verbose=args.debug
                                     )
    elif args.data_list == 'rjh_tswi':
        params = get_default_setting(phase, train_n_batch=args.n_batch, valid_n_batch=args.n_batch, valid_n_workers=5)
        dataset_ = get_rjh_tswi_seg_dataset(files_list,
                                            phase=phase,
                                            crop_size=args.crop_size,
                                            preload=args.preload,
                                            augment_ratio=args.augment_ratio,
                                            cache_dir=check_dir(os.path.dirname(args.experiment_path),'caches'),
                                            verbose=args.debug
                                            )
    else:
        raise ValueError(f'No {args.data_list} dataset')

    loader = DataLoader(dataset_, **params)
    return loader
