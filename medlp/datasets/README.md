# Datasets for MeDLP 

### Here is a simple example for creating a dataset for MeDLP.

    
    from medlp.data_io import CLASSIFICATION_DATASETS

    @CLASSIFICATION_DATASETS.register('my_dataset', '2D', '\homes\my_dataset_fname.json')
    def get_my_dataset(files_list, phase, opts):
        # Get parameters you need for creating your dataset
        preload=opts.get('preload', 1.0)
        image_size=opts.get('image_size', None)
        crop_size=opts.get('crop_size', None)
        augment_ratio=opts.get('augment_ratio', 0.5)

        dataset = SegmentationDataset2D(files_list).get_dataset()

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

    @CLASSIFICATION_DATASETS.register('my_dataset', '2D', '\homes\my_dataset_fname.json')
    @CLASSIFICATION_DATASETS.register('my_dataset_2', '2D', '\homes\my_dataset_fname_2.json')
    def get_my_dataset(files_list, phase, opts):
        pass
