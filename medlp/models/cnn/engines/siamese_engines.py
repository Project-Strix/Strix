import os, re, logging, copy 
from pathlib import Path
import numpy as np
from functools import partial

import torch
from medlp.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from medlp.utilities.utils import assert_network_type, is_avaible_size, output_filename_check
from medlp.models.cnn.utils import output_onehot_transform

from monai_ex.engines import SupervisedTrainer, SupervisedEvaluator, EnsembleEvaluator
from monai_ex.engines import multi_gpu_supervised_trainer
from monai_ex.inferers import SimpleInferer, SlidingWindowClassify, SlidingWindowInferer
from monai_ex.networks import predict_segmentation, one_hot
from monai_ex.utils import Activation, Normalisation
from ignite.metrics import Accuracy, MeanSquaredError, Precision, Recall
from monai_ex.transforms import (
    Compose, 
    ActivationsD, 
    AsDiscreteD, 
    KeepLargestConnectedComponentD, 
    MeanEnsembleD, 
    VoteEnsembleD,
    SqueezeDimD
)

from monai_ex.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    TensorBoardImageHandlerEx,
    ValidationHandler,
    LrScheduleHandler,
    LrScheduleTensorboardHandler,
    CheckpointSaver,
    CheckpointLoader,
    SegmentationSaver,
    ClassificationSaver,
    MeanDice,
    MetricLogger,
    ROCAUC,
)

@TRAIN_ENGINES.register('siamese')
def build_siamese_engine(**kwargs):
    opts = kwargs['opts'] 
    train_loader = kwargs['train_loader']  
    test_loader = kwargs['test_loader'] 
    net = kwargs['net']
    loss = kwargs['loss'] 
    optim = kwargs['optim'] 
    lr_scheduler = kwargs['lr_scheduler'] 
    writer = kwargs['writer'] 
    valid_interval = kwargs['valid_interval'] 
    device = kwargs['device'] 
    model_dir = kwargs['model_dir']
    logger_name = kwargs.get('logger_name', None)

    