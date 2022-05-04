import os, re, logging, copy
from pathlib import Path
import numpy as np
from functools import partial

import torch
from strix.configures import config as cfg
from strix.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from strix.utilities.utils import is_avaible_size, output_filename_check
from strix.models.cnn.engines.utils import output_onehot_transform

from monai_ex.engines import SupervisedTrainer, SupervisedEvaluator, EnsembleEvaluator
from monai_ex.engines import multi_gpu_supervised_trainer
from monai_ex.inferers import SimpleInferer, SlidingWindowClassify, SlidingWindowInferer
from ignite.metrics import MeanSquaredError

from monai_ex.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
    LrScheduleTensorboardHandler,
    TensorBoardImageHandlerEx,
    CheckpointSaver,
)


@TRAIN_ENGINES.register("selflearning")
def build_selflearning_engine(**kwargs):
    opts = kwargs["opts"]
    train_loader = kwargs["train_loader"]
    test_loader = kwargs["test_loader"]
    net = kwargs["net"]
    loss = kwargs["loss"]
    optim = kwargs["optim"]
    lr_scheduler = kwargs["lr_scheduler"]
    writer = kwargs["writer"]
    valid_interval = kwargs["valid_interval"]
    device = kwargs["device"]
    model_dir = kwargs["model_dir"]
    logger_name = kwargs.get("logger_name", None)
    image_ = cfg.get_key("image")
    label_ = cfg.get_key("label")
    pred_ = cfg.get_key("pred")
    loss_ = cfg.get_key("loss")

    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="val_loss"),
        TensorBoardImageHandlerEx(
            summary_writer=writer,
            batch_transform=lambda x: (x[image_], x[label_]),
            output_transform=lambda x: x[pred_],
            max_channels=opts.output_nc,
            prefix_name="Val",
        ),
        CheckpointSaver(
            save_dir=model_dir,
            save_dict={"net": net},
            save_key_metric=True,
            key_metric_name="val_mse",
            key_metric_mode="min",
            key_metric_n_saved=2,
        ),
    ]

    prepare_batch_fn = lambda x, device, nb: (
        x[image_].to(device),
        x[label_].to(device),
    )
    key_metric_transform_fn = lambda x: (x[pred_], x[label_])

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        epoch_length=int(opts.n_epoch_len)
        if opts.n_epoch_len > 1.0
        else int(opts.n_epoch_len * len(test_loader)),
        prepare_batch=prepare_batch_fn,
        inferer=SimpleInferer(),
        key_val_metric={"val_mse": MeanSquaredError(key_metric_transform_fn)},
        val_handlers=val_handlers,
        amp=opts.amp,
    )

    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_step_transform = lambda x: evaluator.state.metrics["val_mse"]
    else:
        lr_step_transform = lambda x: ()

    train_handlers = [
        LrScheduleTensorboardHandler(
            lr_scheduler=lr_scheduler,
            summary_writer=writer,
            step_transform=lr_step_transform,
        ),
        ValidationHandler(
            validator=evaluator, interval=valid_interval, epoch_level=True
        ),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x[loss_]),
        CheckpointSaver(
            save_dir=model_dir,
            save_dict={"net": net, "optim": optim},
            save_interval=opts.save_epoch_freq,
            epoch_level=True,
            n_saved=5,
        ),
        TensorBoardStatsHandler(
            summary_writer=writer,
            tag_name="train_loss",
            output_transform=lambda x: x[loss_],
        ),
        TensorBoardImageHandlerEx(
            summary_writer=writer,
            batch_transform=lambda x: (x[image_], x[label_]),
            output_transform=lambda x: x[pred_],
            max_channels=opts.output_nc,
            prefix_name="train",
        ),
    ]

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
        train_handlers=train_handlers,
        amp=opts.amp,
    )
    return trainer
