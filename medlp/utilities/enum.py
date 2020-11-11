
# All enum variables
DATASET_LIST = ['picc_h5', 'picc_nii', 'picc_dcm', 'Obj_CXR', 'NIH_CXR', 'rib', 'kits', 'jsph_rcc', 'rjh_tswi']
NORM_TYPES = ['batch','instance','group','auto']
LOSSES = ['CE', 'WCE', 'MSE', 'DCE', 'CE-DCE', 'WCE-DCE']
LR_SCHEDULE = ['const', 'lambda', 'step', 'SGDR', 'plateau']
FRAMEWORK_TYPES = ['segmentation','classification','siamese','selflearning','detection']
LAYER_ORDERS = ['crb','cbr', 'cgr','cbe', 'abn']
OPTIM_TYPES = ['sgd', 'adam', 'adamw', 'radam']

CNN_MODEL_TYPES = ['vgg13', 'vgg16', 'resnet34', 'resnet50']
FCN_MODEL_TYPES = ['unet', 'res-unet', 'unetv2', 'res-unetv2', 'scnn', 'highresnet']
RCNN_MODEL_TYPES = ['mask_rcnn', 'faster_rcnn', 'fcos', 'retina']
NETWORK_TYPES = {'CNN':CNN_MODEL_TYPES, 
                 'FCN':FCN_MODEL_TYPES,
                 'RCNN':RCNN_MODEL_TYPES}
RCNN_BACKBONE = ["R-50-C4","R-50-C5","R-101-C4","R-101-C5",
                 "R-50-FPN","R-101-FPN","R-152-FPN"]