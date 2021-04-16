# All enum variables
# DATASET_LIST = ['picc_h5', 'picc_nii', 'picc_dcm', 'Obj_CXR', 'NIH_CXR', 'rib', 'kits', 'jsph_rcc', 'rjh_tswi', 'rjh_swim', 'rjh_tswi_cls']

OUTPUT_DIR = "/homes/clwang/Data/medlp_exp"

DIMS = ["2D", "3D", "2", "3", 2, 3]
PHASES = ["train", "valid", "test"]
NORM_TYPES = ["batch", "instance", "group"]
LOSSES = [
    "CE",
    "WCE",
    "BCE",
    "WBCE",
    "MSE",
    "DCE",
    "GDL",
    "CE-DCE",
    "WCE-DCE",
    "FocalLoss",
    "FocalDiceLoss",
]
LR_SCHEDULE = ["const", "poly", "step", 'multistep', "SGDR", "plateau"]
FRAMEWORK_TYPES = [
    "segmentation",
    "classification",
    "siamese",
    "selflearning",
    "detection",
    "multitask",
]
LAYER_ORDERS = ["crb", "cbr", "cgr", "cbe", "abn"]
OPTIM_TYPES = ["sgd", "adam", "adamw", "radam"]

CNN_MODEL_TYPES = ["vgg9", "vgg13", "vgg16", "resnet18", "resnet34", "resnet50"]
FCN_MODEL_TYPES = [
    "unet",
    "res-unet",
    "unetv2",
    "res-unetv2",
    "scnn",
    "highresnet",
    "vnet",
]
RCNN_MODEL_TYPES = ["mask_rcnn", "faster_rcnn", "fcos", "retina"]
NETWORK_TYPES = {
    "CNN": CNN_MODEL_TYPES,
    "FCN": FCN_MODEL_TYPES,
    "RCNN": RCNN_MODEL_TYPES,
}
RCNN_BACKBONE = [
    "R-50-C4",
    "R-50-C5",
    "R-101-C4",
    "R-101-C5",
    "R-50-FPN",
    "R-101-FPN",
    "R-152-FPN",
]
