from pathlib import Path
import nibabel as nib
from utils_cw import get_items_from_file, check_dir
from strix.data_io.dataio import SEGMENTATION_DATASETS 

dataset_name = 'lidc'
dataset_type = SEGMENTATION_DATASETS['3D'][dataset_name]
dataset_list = SEGMENTATION_DATASETS['3D'][dataset_name+'_fpath']

lidc = dataset_type(
    get_items_from_file(dataset_list),
    'valid',
    {}
)

out_dir = check_dir('/homes/clwang/Data/strix_exp/LIDC_test_crops')
for i, data in enumerate(lidc):
    filename = Path(data['image_meta_dict']['filename_or_obj'])
    print(i, filename)
    nib.save( nib.Nifti1Image(data['image'].numpy().squeeze(), data['image_meta_dict']['affine']), out_dir/f'{i}-{filename.parent.parent.stem}-image.nii.gz' )
    nib.save( nib.Nifti1Image(data['label'].numpy().squeeze(), data['label_meta_dict']['affine']), out_dir/f'{i}-{filename.parent.parent.stem}-label.nii.gz' )
