from monai_ex import data
from medlp.data_io import SIAMESE_DATASETS
from monai_ex.data import DataLoader
from utils_cw import get_items_from_file


dataset = SIAMESE_DATASETS['2D']['jsph_mvi'](
    files_list=get_items_from_file("/homes/clwang/Data/jsph_lung/MVI/data_crops/datalist-train.json"),
    phase='valid', opts={})
loader = DataLoader(dataset=dataset, batch_size=5)

for d in loader:
    print(type(d), len(d))
    print(type(d[0]), len(d[0]))
