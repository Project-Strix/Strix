import os 
from pathlib import Path
import numpy as np
from functools import partial

import torch
from medlp.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from medlp.utilities.utils import assert_network_type, is_avaible_size, output_filename_check
from medlp.models.cnn.utils import output_onehot_transform

from monai_ex.engines import multi_gpu_supervised_trainer
from monai_ex.inferers import SimpleInferer, SlidingWindowClassify
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

from monai_ex.engines import (
    SupervisedTrainer,
    SupervisedEvaluator,
    EnsembleEvaluator
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

    if opts.criterion in ["CE", "WCE"]:
        prepare_batch_fn = lambda x, device, nb: (
            x["image"].to(device),
            x["label"].squeeze(dim=1).to(device),
        )
        if opts.output_nc > 1:
            key_metric_transform_fn = lambda x: (
                x["pred"],
                one_hot(x["label"].unsqueeze(dim=1), num_classes=opts.output_nc),
            )
    elif opts.criterion in ["BCE", "WBCE"]:
        prepare_batch_fn = lambda x, device, nb: (
            x["image"].to(device),
            torch.as_tensor(x["label"], dtype=torch.float32).to(device),
        )
        if opts.output_nc > 1:
            key_metric_transform_fn = lambda x: (
                x["pred"],
                one_hot(x["label"], num_classes=opts.output_nc),
            )
    else:
        prepare_batch_fn = lambda x, device, nb: (
            x["image"].to(device),
            x["label"].to(device),
        )
        if opts.output_nc > 1:
            key_metric_transform_fn = lambda x: (
                x["pred"],
                one_hot(x["label"], num_classes=opts.output_nc),
            )

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=opts.n_epoch,
        train_data_loader=train_loader,
        network=net,
        optimizer=optim,
        loss_function=loss,
        epoch_length=int(opts.n_epoch_len)
        if opts.n_epoch_len > 1.0
        else int(opts.n_epoch_len * len(train_loader)),
        prepare_batch=prepare_batch_fn,
        inferer=SimpleInferer(),
        post_transform=trainval_post_transforms,
        key_train_metric={
            "train_mean_dice": MeanDice(
                include_background=False, output_transform=key_metric_transform_fn
            )
        },
        train_handlers=train_handlers,
        amp=opts.amp,
    )