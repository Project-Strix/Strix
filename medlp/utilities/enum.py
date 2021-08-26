# All enum variables

BUILTIN_TYPES = [dict, list, tuple, str, int, float, bool]

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
    "DiceFocalLoss",
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
