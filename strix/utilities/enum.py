# All enum variables
from enum import Enum

BUILTIN_TYPES = [dict, list, tuple, str, int, float, bool]


def get_enums(enum_class):
    return [item.value for item in enum_class.__members__.values()]


class Dims(Enum):
    TWO = "2D"
    THREE = "3D"


DIMS = get_enums(Dims)


class Phases(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST_IN = "test"
    TEST_EX = "test_wo_label"


PHASES = get_enums(Phases)


class Norms(Enum):
    BATCH = "batch"
    INSTANCE = "instance"
    LAYER = "layer"
    GROUP = "group"


NORMS = get_enums(Norms)


class Activations(Enum):
    RELU = "relu"
    LEAKYRELU = "leakyrelu"
    PRELU = "prelu"
    SELU = "selu"
    CELU = "celu"
    GELU = "gelu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SOFTMAX = "softmax"
    LOGSOFTMAX = "logsoftmax"
    SWISH = "swish"
    MISH = "mish"


ACTIVATIONS = get_enums(Activations)


class Losses(Enum):
    CE = "CE"
    WCE = "WCE"
    BCE = "BCE"
    WBCE = "WBCE"
    MSE = "MSE"
    DCE = "DCE"
    GDL = "GDL"
    CE_DCE = "CE-DCE"
    WCE_DCE = "WCE-DCE"
    FOCAL = "FocalLoss"
    DICE_FOCAL = "DiceFocalLoss"


LOSSES = get_enums(Losses)


class LrSchedule(Enum):
    CONST = "const"
    POLY = "poly"
    STEP = "step"
    MULTISTEP = "multistep"
    SGDR = "SGDR"
    PLATEAU = "plateau"
    EXP = "exponential"
    LINEAR = "linear"


LR_SCHEDULES = get_enums(LrSchedule)


class Frameworks(Enum):
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    # SIAMESE = "siamese"
    SELFLEARNING = "selflearning"
    DETECTION = "detection"
    MULTITASK = "multitask"


FRAMEWORKS = get_enums(Frameworks)


class NetworkArgs(Enum):
    SPATIAL_DIMS = "spatial_dims"
    IN_CHANNELS = "in_channels"
    OUT_CHANNELS = "out_channels"
    ACT = "act"
    NORM = "norm"
    N_DEPTH = "n_depth"
    N_GROUP = "n_group"
    DROP_OUT = "drop_out"
    IS_PRUNABLE = "is_prunable"
    PRETRAINED = "pretrained"
    PRETRAINED_MODEL_PATH = "pretrained_model_path"


NETWORK_ARGS = get_enums(NetworkArgs)


class LayerOrders(Enum):
    CRB = "crb"
    CBR = "cbr"
    CGR = "cgr"
    CBE = "cbe"
    ABN = "abn"


LAYER_ORDERS = get_enums(LayerOrders)


class Optimizers(Enum):
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RADAM = "radam"
    RANGER = "ranger"


OPTIMIZERS = get_enums(Optimizers)


class Freezers(Enum):
    UNTIL = "until"
    AUTO = "auto"
    FULL = "full"
    SUBTASK = "subtask"

FREEZERS = get_enums(Freezers)


class DatalistKeywords(Enum):
    UNLABEL = "unlabeled"
    LABEL = "labeled"

DATALISTKEYWORDS = get_enums(DatalistKeywords)


class SerialFileFormat(Enum):
    JSON = "json"
    YAML = "yaml"
