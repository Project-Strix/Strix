import os
from medlp.data_io.rjh_dataset import get_rjh_tswi_seg_dataset
from medlp.utilities.transforms import LabelMorphology

from utils_cw import get_items_from_file
import nibabel as nib
import numpy as np

print('--dilation--')
a = np.zeros((1, 8, 8))
a[:, 1, 2] = 1
a[:, 5:7,5:7] = 2
print(a)
converter = LabelMorphology('dilation', radius=2, binary=False)
ret = converter(a)
print(ret.shape, '\n', ret)

print('--closing--')
a = np.zeros((1,7,7), dtype=int)
a[:, 1:6, 2:5] = 1
a[:, 1:3,3] = 0

converter = LabelMorphology('closing', radius=1, binary=False)
ret = converter(a)
print(ret.shape, '\n', ret)

