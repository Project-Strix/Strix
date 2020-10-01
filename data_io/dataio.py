import tqdm, math, random, time
from PIL import Image
import inspect, re, os, h5py, collections, json, csv
import numpy as np
from skimage.exposure import rescale_intensity
from utils_cw import Print, load_h5
from data_io.picc_dataset import get_PICC_dataset, get_RIB_dataset, CacheDataset

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

def get_picc_datalist(dataset_name):
    if dataset_name == 'picc_h5':
        if os.name == 'nt':
            fname = r"\\mega\clwang\Data\picc\prepared_h5\data_list.json"
        elif os.name == 'posix':
            fname = "/homes/clwang/Data/picc/prepared_h5/data_list_linux.json"
    elif dataset_name == 'all_dr':
        raise NotImplementedError
    elif dataset_name == 'rib':
        fname = "/homes/clwang/Data/picc/prepared_rib_h5/nii_files.json"
        #fname = "/homes/yliu/Code/picc/raw_data2.json"
    else:
        raise ValueError
    
    return fname

def load_picc_h5_data_once(file_list, h5_keys=['image', 'roi', 'coord'], transpose=None):
    #Pre-load all training data once.
    data = { i:[] for i in h5_keys }
    Print('\nPreload all {} training data'.format(len(file_list)), color='g')
    for fname in tqdm.tqdm(file_list):
        try:
            data_ = load_h5(fname, keywords=h5_keys, transpose=transpose)
            # if ds>1:
            #     data = data[::ds,::ds]
            #     roi  = roi[::ds,::ds]
            #     coord = coord[0]/ds, coord[1]/ds
        except Exception as e:
            Print('Data not exist!', fname, color='r')
            print(e)
            continue

        for i, key in enumerate(h5_keys):
            data[key].append(data_[i])
    return data.values()

def get_dataloader(args, files_list, phase='train'):
    if phase == 'train': #Todo: move this part to each dataset
        shuffle = True
        augment_ratio = args.augment_ratio
        n_batch = args.n_batch
        num_workers = 10
        drop_last = True
    elif phase == 'valid':
        shuffle = True
        augment_ratio = 0.
        n_batch = 3 #math.ceil(args.n_batch/2)
        num_workers = 0
        drop_last = True
    elif phase == 'test':
        shuffle = False
        augment_ratio = 0.
        n_batch = 1
        num_workers = 2
        drop_last = False
    else:
        raise ValueError(f"phase must be in 'train,valid,test', but got {phase}") 

    if args.data_list == 'rib':
        dataset_ = get_RIB_dataset(files_list, phase=phase, in_channels=args.input_nc, preload=args.preload, image_size=args.image_size,
                                   crop_size=args.crop_size, augment_ratio=augment_ratio, downsample=args.downsample, verbose=args.debug)
    elif args.data_list == 'picc_h5':
        dataset_ = get_PICC_dataset(files_list, phase=phase, spacing=[0.3,0.3], in_channels=args.input_nc, crop_size=args.crop_size,
                                    preload=args.preload, augment_ratio=augment_ratio, downsample=args.downsample, verbose=args.debug)

    loader = DataLoader(dataset_, batch_size=n_batch, shuffle=shuffle, 
                        drop_last=drop_last, num_workers=num_workers, pin_memory=True)

    return loader
    