import os, re, logging, copy 
from pathlib import Path
import numpy as np
from functools import partial

import torch
from medlp.utilities.handlers import NNIReporterHandler
from medlp.utilities.utils import (
    ENGINES, 
    TEST_ENGINES, 
    ENSEMBLE_TEST_ENGINES, 
    assert_network_type, 
    is_avaible_size, 
    output_filename_check
)
from medlp.models.cnn.utils import output_onehot_transform

from monai.losses import DiceLoss
from monai.engines import SupervisedTrainer, SupervisedEvaluator, EnsembleEvaluator
from monai.engines import multi_gpu_supervised_trainer
from monai.inferers import SimpleInferer, SlidingWindowClassify, SlidingWindowInferer
from monai.networks import predict_segmentation, one_hot
from monai.utils import Activation, ChannelMatching, Normalisation
from ignite.metrics import Accuracy, MeanSquaredError, Precision, Recall
from monai.transforms import (
    Compose, 
    ActivationsD, 
    AsDiscreteD, 
    KeepLargestConnectedComponentD, 
    MeanEnsembleD, 
    VoteEnsembleD,
    SqueezeDimD
)


from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    MyTensorBoardImageHandler,
    ValidationHandler,
    LrScheduleHandler,
    LrScheduleTensorboardHandler,
    CheckpointSaver,
    CheckpointLoader,
    SegmentationSaver,
    ClassificationSaver,
)


@ENGINES.register('selflearning')
def build_selflearning_engine(**kwargs):
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

    assert_network_type(opts.model_type, 'FCN')

    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="val_loss"),
        MyTensorBoardImageHandler(
            summary_writer=writer, 
            batch_transform=lambda x: (x["image"], x["label"]), 
            output_transform=lambda x: x["pred"],
            max_channels=opts.output_nc,
            prefix_name='Val'
        ),
        CheckpointSaver(save_dir=model_dir, save_dict={"net": net}, save_key_metric=True,
                        key_metric_name='val_mse',key_metric_mode='min',key_metric_n_saved=2)
    ]

    prepare_batch_fn = lambda x : (x["image"], x["label"])
    key_metric_transform_fn = lambda x : (x["pred"], x["label"])

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len*len(test_loader)),
        prepare_batch=prepare_batch_fn,
        inferer=SimpleInferer(),
        key_val_metric={
            "val_mse": MeanSquaredError(key_metric_transform_fn)
        },
        val_handlers=val_handlers,
        amp=opts.amp
    )

    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_step_transform = lambda x : evaluator.state.metrics["val_mse"]
    else:
        lr_step_transform = lambda x: ()

    train_handlers = [
        LrScheduleTensorboardHandler(lr_scheduler=lr_scheduler, summary_writer=writer, step_transform=lr_step_transform),
        ValidationHandler(validator=evaluator, interval=valid_interval, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x:x["loss"]),
        CheckpointSaver(save_dir=model_dir, save_dict={"net":net, "optim":optim}, save_interval=opts.save_epoch_freq, epoch_level=True, n_saved=5),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="train_loss", output_transform=lambda x:x["loss"]),
        MyTensorBoardImageHandler(
            summary_writer=writer, batch_transform=lambda x: (x["image"], x["label"]), 
            output_transform=lambda x: x["pred"],
            max_channels=opts.output_nc,
            prefix_name='train'
        ),
    ]
    
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=opts.n_epoch,
        train_data_loader=train_loader,
        network=net,
        optimizer=optim,
        loss_function=loss,
        epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len*len(train_loader)),
        prepare_batch=prepare_batch_fn,
        inferer=SimpleInferer(),
        train_handlers=train_handlers,
        amp=opts.amp
    )
    return trainer

