# Datasets for MeDLP 

### Here is a simple example for creating a custom dataset for MeDLP using python script.

    
    from medlp.data_io import CLASSIFICATION_DATASETS, BasicClassificationDataset
    from monai_ex.transforms import *

    @CLASSIFICATION_DATASETS.register('2D', 'my_dataset', '\homes\my_dataset_fname.json')
    def get_my_dataset(files_list, phase, opts):
        # Get parameters you need for creating your dataset
        preload=opts.get('preload', 1.0)
        augment_ratio=opts.get('augment_ratio', 0.5)

        dataset = BasicClassificationDataset(
            files_list,
            loader = LoadImageD(keys="image"),
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
        )

        return dataset

In the given example, `my_dataset` has been registered to a `2D classification` MeDLP dataset. Major steps to implement a MeDLP dataset should be:

1. import base dataset type `CLASSIFICATION_DATASETS`/`SEGMENTATION_DATASETS`/`SELFLEARNING_DATASETS`/`MULTITASK_DATASETS`.
2. Define your dataset function, with 3 arguments:
    - files_list: handle your input data list
    - phase: including 4 different stages `Phases.TRAIN`, `Phases.VALID`, `Phases.TEST_IN` and `Phases.TEST_EX`
      - `Phases` is imported from `medlp.utilities.enum import Phases`
      - `Phases.TEST_IN` means internal test w/ ground-truth, `Phases.TEST_EX` means external test w/o gt.
    - opts: including all options from input arguments. 
3. Add decorator to your function like the given example. Dataset decorator should have 3 arguments: 
    1. name of your dataset name, ex. "my_dataset1"
    2. dimension of your dataset, must be "2D" or "3D".
    3. file path (.json/.yaml) indicates the location of your training data
    4. [Optional] file path indicates the location of your test data


## __Advanced__
- ### If you want reuse your dataset function for multiple dataset but with different training data. You can add multiple decorators like:
        @CLASSIFICATION_DATASETS.register('2D', 'my_dataset', '/homes/my_dataset_fname.json')
        @CLASSIFICATION_DATASETS.register('2D', 'my_dataset_2', '/homes/my_dataset_fname_2.json')
        def get_my_dataset(files_list, phase, opts):
            pass


- ### Notice that ONLY registered dataset can used in MeDLP. If you want remove one dataset, you can simply remove its decorator instead.
        # @CLASSIFICATION_DATASETS.register('2D', 'my_dataset', '/homes/my_dataset_fname.json') <- Comment this!
        def get_my_dataset(files_list, phase, opts):
            print("Not used")

- ### If you have want to save/snapshot your dataset. You can use `snapshot` decorator! Your `my_dataset.py` will be automatically saved to your experimental dir.
        @CLASSIFICATION_DATASETS.snapshot
        @CLASSIFICATION_DATASETS.register('2D', 'my_dataset', '/homes/my_dataset_fname.json')
        def get_my_dataset(files_list, phase, opts):
            pass

- ### If you have many related datasets designed for same project. You can use `project` decorator! One additional folder will be created for your project under experimental dir.
        @CLASSIFICATION_DATASETS.snapshot
        @CLASSIFICATION_DATASETS.project('Your_project_name')
        @CLASSIFICATION_DATASETS.register('2D', 'my_dataset', '/homes/my_dataset_fname.json')
        def get_my_dataset(files_list, phase, opts):
            pass

- ### MeDLP support multiple inputs/outputs now! Use `multi_in` & `multi_out` decorators. Make sure the keys are in the filelist (.json/.yaml).
        @CLASSIFICATION_DATASETS.snapshot
        @CLASSIFICATION_DATASETS.multi_in('image', 'mask')
        @CLASSIFICATION_DATASETS.register('2D', 'my_dataset', '/homes/my_dataset_fname.json')
        def get_my_dataset(files_list, phase, opts):
            pass
