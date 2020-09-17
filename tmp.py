#%% 
import os
import numpy as np
from monai.transforms import CenterSpatialCropd
from utils_cw import load_h5

fname = r"\\mega\clwang\Data\picc\prepared_h5\1.2.392.200046.100.2.1.154636604036.190318093233.2\1.2.392.200046.100.2.1.154636604036.190318093233.2.h5"
data = load_h5(fname, keywords=["image"])[0]

cropper = CenterSpatialCropd(keys=["image"], roi_size=(2400,2400), allow_pad=True)
d = cropper({"image":data[np.newaxis,...]})
# %%
