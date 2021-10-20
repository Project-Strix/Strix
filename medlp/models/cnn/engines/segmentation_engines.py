import os
import re
import logging
import copy
from pathlib import Path
from functools import partial

import torch
from medlp.utilities.handlers import NNIReporterHandler, TensorboardDumper
from medlp.models.cnn.engines import (
    TRAIN_ENGINES,
    TEST_ENGINES,
    ENSEMBLE_TEST_ENGINES
)
from medlp.models.cnn.engines.utils import (
    get_prepare_batch_fn,
    get_unsupervised_prepare_batch_fn,
    get_dice_metric_transform_fn,
)
from medlp.utilities.utils import (
    is_avaible_size,
    output_filename_check,
    get_attr_
)
from medlp.configures import config as cfg
from medlp.models.cnn.engines.utils import get_models

from monai_ex.inferers import (
    SimpleInferer,
    SlidingWindowInferer,
    SlidingWindowInferer2Dfor3D
)
from monai_ex.networks import one_hot
from ignite.engine import Events
from ignite.handlers import EarlyStopping

from monai_ex.engines import (
    SupervisedTrainer,
    SupervisedEvaluator,
    EnsembleEvaluator
)

from monai_ex.transforms import (
    Compose,
    ActivationsD,
    AsDiscreteD,
    MeanEnsembleD
)
from monai_ex.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandlerEx,
    ValidationHandler,
    LrScheduleTensorboardHandler,
    CheckpointSaverEx,
    CheckpointLoader,
    SegmentationSaver,
    MeanDice,
    ROCAUC,
    stopping_fn_from_metric,
)


@TRAIN_ENGINES.register("segmentation")
def build_segmentation_engine(**kwargs):
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
    multi_input_keys = kwargs.get('multi_input_keys', None)
    multi_output_keys = kwargs.get('multi_output_keys', None)
    _image_ = cfg.get_key("image")
    _label_ = cfg.get_key("label")
    _pred_ = cfg.get_key("pred")
    _loss_ = cfg.get_key("loss")

    val_metric = "val_mean_dice"
    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        TensorBoardStatsHandler(summary_writer=writer, tag_name=val_metric),
        TensorBoardImageHandlerEx(
            summary_writer=writer,
            batch_transform=lambda x: (x[_image_], x[_label_]),
            output_transform=lambda x: x[_pred_],
            max_channels=opts.output_nc,
            prefix_name="Val",
        ),
    ]

    # save N best model handler
    if opts.save_n_best > 0:
        val_handlers += [
            CheckpointSaverEx(
                save_dir=model_dir/"Best_Models",
                save_dict={"net": net},
                file_prefix=val_metric,
                save_key_metric=True,
                key_metric_n_saved=opts.save_n_best,
                key_metric_save_after_epoch=0
            ),
            TensorboardDumper(
                log_dir=writer.log_dir,
                epoch_level=True,
                logger_name=logger_name,
            )
        ]

    # If in nni search mode
    if opts.nni:
        val_handlers += [
            NNIReporterHandler(
                metric_name=val_metric,
                max_epochs=opts.n_epoch,
                logger_name=logger_name
            )
        ]

    if opts.output_nc == 1:
        trainval_post_transforms = Compose(
            [
                ActivationsD(keys=_pred_, sigmoid=True),
                AsDiscreteD(keys=_pred_, threshold_values=True, logit_thresh=0.5),
            ]
        )
    else:
        trainval_post_transforms = Compose(
            [
                ActivationsD(keys=_pred_, softmax=True),
                AsDiscreteD(
                    keys=_pred_, to_onehot=True, argmax=True, n_classes=opts.output_nc
                ),
                # KeepLargestConnectedComponentD(keys=_pred_, applied_labels=[1], independent=False),
            ]
        )

    prepare_batch_fn = get_prepare_batch_fn(
        opts, _image_, _label_, multi_input_keys, multi_output_keys
    )

    dice_metric_transform_fn = get_dice_metric_transform_fn(
        opts, pred_key=_pred_, label_key=_label_
    )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        epoch_length=int(opts.n_epoch_len)
        if opts.n_epoch_len > 1.0
        else int(opts.n_epoch_len * len(test_loader)),
        prepare_batch=prepare_batch_fn,
        inferer=SimpleInferer(),  # SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        post_transform=trainval_post_transforms,
        key_val_metric={
            val_metric: MeanDice(
                include_background=False,
                output_transform=dice_metric_transform_fn
            )
        },
        val_handlers=val_handlers,
        amp=opts.amp,
    )

    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_step_transform = lambda x: evaluator.state.metrics[val_metric]
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
        StatsHandler(
            tag_name="train_loss",
            output_transform=lambda x: x[_loss_],
            name=logger_name,
        ),
        CheckpointSaverEx(
            save_dir=model_dir/"Checkpoint",
            save_dict={"net": net, "optim": optim},
            save_interval=opts.save_epoch_freq,
            epoch_level=True,
            n_saved=opts.save_n_best
        ),
        TensorBoardStatsHandler(
            summary_writer=writer,
            tag_name="train_loss",
            output_transform=lambda x: x[_loss_],
        ),
        TensorBoardImageHandlerEx(
            summary_writer=writer,
            batch_transform=lambda x: (x[_image_], x[_label_]),
            output_transform=lambda x: x[_pred_],
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
        post_transform=trainval_post_transforms,
        key_train_metric={
            "train_mean_dice": MeanDice(
                include_background=False,
                output_transform=dice_metric_transform_fn
            )
        },
        train_handlers=train_handlers,
        amp=opts.amp,
    )

    if opts.early_stop > 0:
        early_stopper = EarlyStopping(
            patience=opts.early_stop,
            score_function=stopping_fn_from_metric(val_metric),
            trainer=trainer,
        )
        evaluator.add_event_handler(
            event_name=Events.EPOCH_COMPLETED, handler=early_stopper
        )
    return trainer


@TEST_ENGINES.register("segmentation")
def build_segmentation_test_engine(**kwargs):
    opts = kwargs["opts"]
    test_loader = kwargs["test_loader"]
    net = kwargs["net"]
    device = kwargs["device"]
    logger_name = kwargs.get("logger_name", None)
    crop_size = get_attr_(opts, 'crop_size', None)
    n_batch = opts.n_batch
    resample = opts.resample
    use_slidingwindow = opts.slidingwindow
    is_multilabel = opts.output_nc > 1
    is_supervised = opts.phase == 'test'
    multi_input_keys = kwargs.get('multi_input_keys', None)
    multi_output_keys = kwargs.get('multi_output_keys', None)
    _image_ = cfg.get_key("image")
    _pred_ = cfg.get_key("pred")
    _label_ = cfg.get_key("label")

    if use_slidingwindow:
        print("---Use slidingwindow infer!---")
        print('patch size:', crop_size)
    else:
        print("---Use simple infer!---")

    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        CheckpointLoader(load_path=opts.model_path, load_dict={"net": net}),
        SegmentationSaver(
            output_dir=opts.out_dir,
            output_ext=".nii.gz",
            resample=resample,
            data_root_dir=output_filename_check(test_loader.dataset),
            batch_transform=lambda x: x[_image_+"_meta_dict"],
            output_transform=lambda output: output[_pred_],
        ),
    ]

    if opts.save_image:
        val_handlers += [
            SegmentationSaver(
                output_dir=opts.out_dir,
                output_postfix=_image_,
                output_ext=".nii.gz",
                resample=resample,
                data_root_dir=output_filename_check(test_loader.dataset),
                batch_transform=lambda x: x[_image_+"_meta_dict"],
                output_transform=lambda output: output[_image_],
            )
        ]

    if is_supervised:
        prepare_batch_fn = get_prepare_batch_fn(
            opts, _image_, _label_, multi_input_keys, multi_output_keys
        )
    else:
        prepare_batch_fn = get_unsupervised_prepare_batch_fn(
            opts, _image_, multi_input_keys
        )

    def onehot_transform(output, n_classes=3, dim=1):
        if n_classes == 1:
            return output
        y_pred, y = output[_pred_], output[_label_]
        return one_hot(y_pred, n_classes, dim=dim), one_hot(y, n_classes, dim=dim)

    if is_multilabel:
        dce_post_transforms = Compose(
            [
                ActivationsD(keys=_pred_, softmax=True),
                AsDiscreteD(keys=_pred_, argmax=True, to_onehot=False, n_classes=opts.output_nc),
                partial(onehot_transform, n_classes=opts.output_nc),
            ]
        )
    else:
        dce_post_transforms = Compose(
            [
                ActivationsD(keys=_pred_, sigmoid=True),
                AsDiscreteD(keys=_pred_, threshold_values=True, logit_thresh=0.5),
                partial(onehot_transform, n_classes=opts.output_nc),
            ]
        )

    if opts.phase == "test_wo_label":
        key_val_metric = None
    elif opts.phase == "test":
        key_val_metric = {
            "val_mean_dice": MeanDice(
                include_background=False, output_transform=dce_post_transforms
            )
        }

    if use_slidingwindow:
        if opts.tensor_dim == '2D':
            inferer = SlidingWindowInferer2Dfor3D(roi_size=crop_size, sw_batch_size=n_batch, overlap=0.6)
        elif opts.tensor_dim == '3D':
            inferer = SlidingWindowInferer(roi_size=crop_size, sw_batch_size=n_batch, overlap=0.5)
    else:
        inferer = SimpleInferer()

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        prepare_batch=prepare_batch_fn,
        inferer=inferer,
        post_transform=None,
        key_val_metric=key_val_metric,
        val_handlers=val_handlers,
        amp=opts.amp,
    )

    return evaluator


@ENSEMBLE_TEST_ENGINES.register("segmentation")
def build_segmentation_ensemble_test_engine(**kwargs):
    raise NotImplementedError("Not finished!")

    opts = kwargs["opts"]
    test_loader = kwargs["test_loader"]
    net = kwargs["net"]
    device = kwargs["device"]
    best_model = kwargs.get("best_val_model", True)
    logger_name = kwargs.get("logger_name", None)
    logger = logging.getLogger(logger_name)
    is_multilabel = opts.output_nc > 1
    use_slidingwindow = is_avaible_size(opts.crop_size)
    float_regex = r"=(-?\d+\.\d+).pt"

    cv_folders = [Path(opts.experiment_path) / f"{i}-th" for i in range(opts.n_fold)]
    cv_folders = filter(lambda x: x.is_dir(), cv_folders)
    best_models = get_models(cv_folders, 'best' if best_model else 'last')
    best_models = list(filter(lambda x: x is not None and x.is_file(), best_models))

    if len(best_models) != opts.n_fold:
        print(
            f"Found {len(best_models)} best models,"
            f"not equal to {opts.n_fold} n_folds.\n"
            f"Use {len(best_models)} best models"
        )
    print(f"Using models: {[m.name for m in best_models]}")

    nets = [copy.deepcopy(net),] * len(best_models)
    for net, m in zip(nets, best_models):
        CheckpointLoader(load_path=str(m), load_dict={"net": net}, name=logger_name)(None)

    pred_keys = [f"pred{i}" for i in range(len(best_models))]

    if best_model:
        w_ = [float(re.search(float_regex, m.name).group(1)) for m in best_models]
    else:
        w_ = None

    post_transforms = MeanEnsembleD(
        keys=pred_keys,
        output_key="pred",
        # in this particular example, we use validation metrics as weights
        weights=w_,
    )

    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=test_loader,
        pred_keys=pred_keys,
        networks=nets,
        inferer=SimpleInferer(),
        post_transform=post_transforms,
        val_handlers=val_handlers,
        key_val_metric={"test_acc": Accuracy(output_transform=acc_post_transforms,is_multilabel=is_multilabel)},
        additional_metrics={
            'test_auc':ROCAUC(output_transform=auc_post_transforms), 
            'Prec':Precision(output_transform=acc_post_transforms),
            'Recall':Recall(output_transform=acc_post_transforms)
        },
    )

    return evaluator
