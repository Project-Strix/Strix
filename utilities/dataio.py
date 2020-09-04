import tqdm, math
from PIL import Image
import inspect, re, os, h5py, collections, json, csv
import numpy as np
from torch.utils.data import DataLoader
from skimage.exposure import rescale_intensity
from utils_cw import Print, load_h5
from utilities.picc_dataset import RandomCropDataset

def get_picc_datalist(dataset_name):
    if dataset_name == 'picc_h5':
        if os.name == 'nt':
            fname = r"\\mega\clwang\Data\picc\prepared_h5\data_list.json"
        elif os.name == 'posix':
            fname = "/homes/clwang/Data/picc/prepared_h5/data_list_linux.json"
    elif dataset_name == 'all_dr':
        raise NotImplementedError
    else:
        raise ValueError
    
    return fname

def load_picc_data_once(file_list, ds=1, transpose=None):
    #Pre-load all training data once.
    all_data, all_label, all_coords = [], [], []
    Print('\nPreload all {} training data'.format(len(file_list)), color='g')
    for fname in tqdm.tqdm(file_list):
        try:
            data, roi, coord = load_h5(fname, keywords=['image', 'roi', 'coord'], transpose=transpose)
            if ds>1:
                data = data[::ds,::ds]
                roi  = roi[::ds,::ds]
                coord = coord[0]/ds, coord[1]/ds
        except Exception as e:
            Print('Data not exist!', fname, color='r')
            print(e)
            continue

        all_data.append(data)
        all_label.append(roi)
        all_coords.append(tuple(coord))
    return all_data, all_label, all_coords


def get_dataloader(args, files_list, dataset_type='train', random_crop_type='balance'):
    #Pre-load all training data once.
    if args.preload:
        all_data, all_roi, all_coord = load_picc_data_once(files_list, ds=args.downsample, transpose=args.transpose)
        #all_test_data, all_test_roi, all_test_coord    = load_picc_data_once(files_test, ds=args.downsample, transpose=args.transpose)
    else:
        all_data = all_roi = all_coord = files_list
    
    shuffle = True if dataset_type == 'train' else False
    augment_ratio = args.augment_ratio if dataset_type == 'train' else 0.0
    n_batch = args.n_batch if dataset_type == 'train' else math.ceil(args.n_batch/4)
    picc_dataset = RandomCropDataset(all_data, all_roi, all_coord, n_classes=3, 
                                     augment_ratio=augment_ratio, crop_size=args.crop_size,
                                     downsample=args.downsample, random_type=random_crop_type,
                                     verbose=args.debug)
    loader = DataLoader(picc_dataset, batch_size=n_batch, shuffle=shuffle, drop_last=True, num_workers=10, pin_memory=True)

    return loader
    