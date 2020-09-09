import tqdm, math, random, time
from PIL import Image
import inspect, re, os, h5py, collections, json, csv
import numpy as np
from torch.utils.data import DataLoader, get_worker_info
from skimage.exposure import rescale_intensity
from utils_cw import Print, load_h5
from utilities.picc_dataset import RandomCropDataset, Rib_dataset, CacheDataset

from monai.transforms import (
        Compose,
        LoadHdf5d,
        LoadNumpyd,
        AddChanneld,
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
        fname = "/homes/clwang/Data/picc/prepared_rib_h5/data_list.json"
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

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id + int(time.time() * 1000 % 1000))

def get_dataloader(args, files_list, dataset_type='train', random_crop_type='balance'):

    if dataset_type == 'train':
        shuffle = True
        augment_ratio = args.augment_ratio
        n_batch = args.n_batch
        num_workers = 10
        drop_last = True
    elif dataset_type == 'valid':
        shuffle = True
        augment_ratio = 0.
        n_batch = math.ceil(args.n_batch/4)
        num_workers = 1
        drop_last = True
    elif dataset_type == 'test':
        shuffle = False
        augment_ratio = 0.
        n_batch = 1
        num_workers = 2
        drop_last = False
    else:
        raise ValueError(f"dataset_type must be in 'train,valid,test', but got {dataset_type}") 

    if args.data_list == 'rib':
        if args.preload:
            all_data, all_roi = load_picc_h5_data_once(files_list, h5_keys=['image', 'roi'], transpose=args.transpose)
        else:
            all_data = all_roi = all_coord = files_list

        dataset_ = Rib_dataset(all_data, all_roi, mode=dataset_type, augmentation_prob=augment_ratio)
    elif args.data_list == 'picc_h5':
        if args.preload:
            all_data, all_roi, all_coord = load_picc_h5_data_once(files_list, h5_keys=['image', 'roi', 'coord'], transpose=args.transpose)
        else:
            all_data = all_roi = all_coord = files_list

        if dataset_type == 'train' or dataset_type == 'valid':
            dataset_ = RandomCropDataset(all_data, all_roi, all_coord, n_classes=3, 
                                         augment_ratio=augment_ratio, crop_size=args.crop_size,
                                         downsample=args.downsample, random_type=random_crop_type,
                                         verbose=args.debug)
        elif dataset_type == 'test':
            if args.preload:
                data_reader = LoadNumpyd(keys=["image","label"])
            else:
                data_reader = LoadHdf5d(keys=["image","label","affine"], h5_keys=["image","roi","affine"])

            test_transforms = Compose(
                [   
                    data_reader,    
                    AddChanneld(keys=["image", "label"]),
                    ToTensord(keys=["image", "label"]),
                ]
            )
            dataset_ = CacheDataset(all_data, transform=test_transforms, cache_rate=0)
    
    loader = DataLoader(dataset_, batch_size=n_batch, shuffle=shuffle, drop_last=drop_last, 
                        num_workers=num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    return loader
    