#%% 
import os, json
import numpy as np
import monai
import monai
from monai.utils import Method, NumpyPadMode, ensure_tuple, ensure_tuple_rep, fall_back_tuple
import matplotlib.pyplot as plt
from monai.transforms import (
    AddChanneld,
    Compose,
    SpatialCrop, 
    RandCropByPosNegLabeld,
    RandRotate90d,
    RandFlipd,
    RandRotated,
    RandScaleIntensityd,
    Resized,
    ToTensord,
    MapTransform, 
    Randomizable
)
from utilities import PICC_RandCropByPosNegLabeld
from utils_cw import load_h5, recursive_glob, Print

#%%
from utilities.picc_dataset import RandomCropDataset
from torch.utils.data import DataLoader

if __name__ == "__main__":
    js_files = r"\\mega\clwang\Data\picc\prepared_h5\data_list.json"
    with open(js_files, 'r') as f:
        h5_files = json.load(f)
    all_data = all_roi = all_coord = h5_files

    picc_dataset = RandomCropDataset(all_data, all_roi, all_coord, n_classes=3, 
                                    augment_ratio=0.3, crop_size=(72,72),
                                    downsample=1, verbose=True)
    loader = DataLoader(picc_dataset, batch_size=2, shuffle=False, drop_last=True, num_workers=1, pin_memory=False)

    data = monai.utils.misc.first(loader)
    print(data[0].shape, data[1].shape)
    os.sys.exit()

#%%
if __name__ == "__main211__":

    augmentations = Compose( [
        AddChanneld(keys=["img", "roi"]),
        RandScaleIntensityd(keys="img",factors=(-0.01,0.01), prob=0.9),
        PICC_RandCropByPosNegLabeld(
            keys=["img", "roi"], label_key="roi", tip_key="coord", 
            spatial_size=[72,72], tip=0.3, neg=0.5, num_samples=1
        ),
        RandRotated(keys=["img","roi"], range_x=10, range_y=10, prob=0.3),
        RandFlipd(keys=["img","roi"], prob=0.3, spatial_axis=[0,1]),
        RandRotate90d(keys=["img", "roi"], prob=0.3, spatial_axes=[0,1]),
        ToTensord(keys=["img", "roi"]),
    ])

    root_dir = r'\\mega\clwang\Data\picc\prepared_h5'
    #h5_files = recursive_glob(root_dir, '*.h5')
    js_files = r"\\mega\clwang\Data\picc\prepared_h5\data_list.json"
    with open(js_files, 'r') as f:
        h5_files = json.load(f)
    for file in h5_files:
        image, mask, coord = load_h5(file, keywords=['image', 'roi', 'coord'])
        data = augmentations({"img":image, "roi":mask, "coord":coord})
        print("File:", os.path.basename(file))
        print(type(data), data)
        
        os.sys.exit()
        fig, ax1 = plt.subplots(1,1)
        ax1.imshow(np.squeeze(data[0]['img']), cmap=plt.cm.gray)
        #ax1.set_title(data[0]['crop_label'])
        plt.savefig( os.path.join(root_dir, 'tip_crops', f'{os.path.basename(file)}'.replace('.h5','.jpg')), facecolor="white", edgecolor="none")
        #plt.show()
# %%
# import os, click

# @click.command('test')
# @click.option('--debug', type=bool, default=False)
# def test(debug, **args):
#     print('debug:', debug)

# if __name__ == '__main__':
#     CONTEXT_SETTINGS = dict(
        
#     )
#     test(default_map={'debug': True})

# %%
