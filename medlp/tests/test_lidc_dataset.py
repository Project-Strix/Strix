import json
from medlp.data_io import DATASET_MAPPING


json_file = "/homes/clwang/Data/LIDC-IDRI-Crops/train_datalist_7-3_cls.json"
with open(json_file) as f:
    files_list = json.load(f)

opts = {}
dataset = DATASET_MAPPING['classification']['2D']['lidc-73'](
    files_list, "train", opts
)

for d in dataset:
    print("shape:", d['image'].shape, d['mask'].shape,)
    print('label:', d['label'])
