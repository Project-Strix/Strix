# Datasets for MeDLP 

## Here is a simple example for creating a custom dataset for MeDLP using python script.

    
    from medlp.data_io import CLASSIFICATION_DATASETS, BasicClassificationDataset
    from monai_ex.transforms import *

    @CLASSIFICATION_DATASETS.register('2D', 'my_dataset', '\homes\my_dataset_fname.json')
    def get_my_dataset(files_list, phase, opts):
        # Get parameters you need for creating your dataset
        preload=opts.get('preload', 1.0)
        augment_ratio=opts.get('augment_ratio', 0.5)

        dataset = BasicClassificationDataset(
            files_list,
            loader = LoadNiftiD(keys="image"),
            channeler = AddChannelD(keys="image"),
            orienter = OrientationD(keys="image", axcodes="LPI"),
            spacer = SpacingD(keys="image", pixdim=(0.1, 0.1)),
            rescaler = NormalizeIntensityD(keys="image"),
            resizer = ResizeD(keys="image", spatial_size=(512, 512)),
            cropper = RandSpatialCropD(keys="image", roi_size=(256, 256), random_size=False),
            caster = CastToTypeD(keys=["image", "label"], dtype=[np.float32, np.int64]),
            to_tensor = ToTensorD(keys=["image", "label"]),
            dataset_type = CacheDataset,
            dataset_kwargs = {'cache_rate': 1.0},
            additional_transforms = None,
        ).get_dataset()

        return dataset

In the given example, `my_dataset` has been registered to a `2D classification` MeDLP dataset. Major steps to implement a MeDLP dataset should be:

1. import base dataset type `CLASSIFICATION_DATASETS`/`SEGMENTATION_DATASETS`/`SELFLEARNING_DATASETS`/`MULTITASK_DATASETS`.
2. Define your dataset function, with 3 arguments:
    - files_list: handle your input data list
    - phase: including 3 different stages `train`, `valid` and `test`
    - opts: including all options from input arguments. 
3. Add decorator to your function like the given example. Dataset decorator should have 3 arguments: 
    - name of your dataset name
    - dimension of your dataset
    - file path (.json) indicates the location of your data


### If you want reuse your dataset function for multiple dataset. You can add multiple decorators like:

    @CLASSIFICATION_DATASETS.register('2D', 'my_dataset', '\homes\my_dataset_fname.json')
    @CLASSIFICATION_DATASETS.register('2D', 'my_dataset_2', '\homes\my_dataset_fname_2.json')
    def get_my_dataset(files_list, phase, opts):
        pass


### Notice that ONLY registered dataset can used in MeDLP. If you want remove one dataset, you can simply remove its decorator instead.