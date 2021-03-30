import os
from functools import partial
from typing import KeysView
from ignite import engine
from monai_ex import handlers

import torch
from medlp.models.cnn.engines import (
    TRAIN_ENGINES,
    TEST_ENGINES,
    ENSEMBLE_TEST_ENGINES
)
from medlp.models.cnn.utils import output_onehot_transform
from medlp.models.cnn.losses.losses import ContrastiveLoss

from monai_ex.engines import multi_gpu_supervised_trainer
from monai_ex.inferers import SimpleInferer, SlidingWindowClassify
from monai_ex.networks import predict_segmentation, one_hot
from monai_ex.utils import Activation, Normalisation
from monai_ex.engines.utils import CustomKeys as Keys
from ignite.metrics import Accuracy, MeanSquaredError, Precision, Recall
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.engine import Events

from monai_ex.transforms import (
    Compose,
    ActivationsD,
    AsDiscreteD,
    SqueezeDimD,
)

from monai_ex.engines import (
    SiameseTrainer,
    SiameseEvaluator,
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
    CheckpointSaverEx,
    stopping_fn_from_metric
)


@TRAIN_ENGINES.register("siamese")
def build_siamese_engine(**kwargs):
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

    prepare_batch_fn = lambda x, device, nb: (
        x[Keys.IMAGE].to(device),
        x[Keys.LABEL].to(device),
    )

    train_metric_name = 'train_acc'
    val_metric_name = 'val_acc'
    key_val_metric = Accuracy(
        output_transform=partial(output_onehot_transform, n_classes=opts.output_nc),
        is_multilabel=opts.output_nc > 1,
    )

    val_handlers = [
        StatsHandler(
            output_transform=lambda x: None,
            name=logger_name
        ),
        TensorBoardStatsHandler(
            summary_writer=writer,
            tag_name="val_loss",
            output_transform=lambda x: x[Keys.LOSS]
        ),
    ]
    if isinstance(loss, ContrastiveLoss):
        train_post_transforms = None
    elif opts.output_nc == 1:
        train_post_transforms = ActivationsD(keys=Keys.PRED, sigmoid=True)
    else:
        train_post_transforms = Compose([
            ActivationsD(keys=Keys.PRED, softmax=True),
            AsDiscreteD(keys=Keys.PRED, argmax=True, to_onehot=False),
            SqueezeDimD(keys=Keys.PRED)
        ])

    evaluator = SiameseEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        epoch_length=int(opts.n_epoch_len)
        if opts.n_epoch_len > 1.0
        else int(opts.n_epoch_len*len(test_loader)),
        inferer=SimpleInferer(),
        post_transform=train_post_transforms,
        key_val_metric=None,  # {val_metric_name: key_val_metric},
        val_handlers=val_handlers,
        amp=opts.amp
    )

    if isinstance(loss, ContrastiveLoss):
        save_handler = ModelCheckpoint(
            dirname=model_dir/'Best_Models',
            filename_prefix='Best',
            n_saved=2,
            score_function=lambda x: -x[Keys.LOSS],
            score_name="val_loss"
        )
        evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), save_handler, {'net': net})
    else:
        pass
        # raise NotImplementedError

    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_step_transform = lambda x: evaluator.state.metrics[val_metric_name]
    else:
        lr_step_transform = lambda x: ()

    train_handlers = [
        LrScheduleTensorboardHandler(
            lr_scheduler=lr_scheduler,
            summary_writer=writer,
            step_transform=lr_step_transform
        ),
        StatsHandler(
            tag_name="train_loss",
            output_transform=lambda x: x[Keys.LOSS],
            name=logger_name
        ),
        TensorBoardStatsHandler(
            summary_writer=writer,
            tag_name="train_loss",
            output_transform=lambda x: x[Keys.LOSS]
        ),
        CheckpointSaverEx(
            save_dir=os.path.join(model_dir, "Checkpoint"),
            save_dict={"net": net, "optim": optim},
            save_interval=opts.save_epoch_freq,
            epoch_level=True,
            n_saved=3,
        )
    ]

    trainer = SiameseTrainer(
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
        post_transform=train_post_transforms,
        key_train_metric=None,  # {train_metric_name: key_val_metric},
        train_handlers=train_handlers,
        amp=opts.amp,
    )

    # if opts.early_stop > 0:
    #     early_stopper = EarlyStopping(
    #         patience=opts.early_stop,
    #         score_function=stopping_fn_from_metric(val_metric_name),
    #         trainer=trainer,
    #     )
    #     evaluator.add_event_handler(
    #         event_name=Events.EPOCH_COMPLETED, handler=early_stopper
    #     )

    return trainer
