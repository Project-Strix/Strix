import os
import re
import logging
import copy
from pathlib import Path
from functools import partial

import torch
from medlp.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from medlp.utilities.utils import output_filename_check, get_attr_
from medlp.utilities.enum import Phases
from medlp.models.cnn.utils import output_onehot_transform, onehot_process
from medlp.models.cnn.engines.utils import get_prepare_batch_fn, get_unsupervised_prepare_batch_fn
from medlp.configures import config as cfg

from monai_ex.inferers import SimpleInfererEx
from monai_ex.networks import one_hot
from monai_ex.metrics import DrawRocCurve
from ignite.engine import Events
from ignite.metrics import Accuracy, Precision, Recall
from ignite.handlers import EarlyStopping

from monai_ex.engines import (
    SupervisedTrainerEx,
    SupervisedEvaluator,
    SupervisedEvaluatorEx,
    EnsembleEvaluator,
)

from monai_ex.transforms import (
    ComposeEx as Compose,
    ActivationsD,
    AsDiscreteD,
    MeanEnsembleD,
    VoteEnsembleD,
    SqueezeDimD,
    EnsureTypeD,
)

from monai_ex.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandlerEx,
    ValidationHandler,
    LrScheduleTensorboardHandler,
    CheckpointSaverEx,
    CheckpointLoader,
    CheckpointLoaderEx,
    SegmentationSaver,
    ClassificationSaverEx,
    ROCAUC,
    stopping_fn_from_metric,
    NNIReporterHandler,
    TensorboardDumper,
    LatentCodeSaver,
    from_engine_ex as from_engine,
)


@TRAIN_ENGINES.register("classification")
def build_classification_engine(**kwargs):
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
    decollate = True
    is_multilabel = opts.output_nc > 1
    multi_input_keys = kwargs.get("multi_input_keys", None)
    multi_output_keys = kwargs.get("multi_output_keys", None)
    _image = cfg.get_key("image")
    _label = cfg.get_key("label")
    _pred = cfg.get_key("pred")
    _loss = cfg.get_key("loss")

    prepare_batch_fn = get_prepare_batch_fn(
        opts, _image, _label, multi_input_keys, multi_output_keys
    )

    if is_multilabel:
        val_metric_name = "val_acc"
    else:
        val_metric_name = "val_auc"

    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        TensorBoardStatsHandler(
            summary_writer=writer,
            tag_name=val_metric_name,
            output_transform=lambda x: None,
        ),
        TensorBoardImageHandlerEx(
            summary_writer=writer,
            batch_transform=lambda x: (None, None),
            output_transform=from_engine(_image),
            max_channels=1,
            prefix_name="Val",
        ),
        TensorboardDumper(
            log_dir=writer.log_dir,
            epoch_level=True,
            logger_name=logger_name,
        ),
    ]

    # save N best model handler
    if opts.save_n_best > 0:
        val_handlers += [
            CheckpointSaverEx(
                save_dir=model_dir / "Best_Models",
                save_dict={"net": net},
                file_prefix=val_metric_name,
                save_key_metric=True,
                key_metric_n_saved=opts.save_n_best,
                key_metric_save_after_epoch=0,
            )
        ]

    # If in nni search mode
    if opts.nni:
        val_handlers += [
            NNIReporterHandler(
                metric_name=val_metric_name,
                max_epochs=opts.n_epoch,
                logger_name=logger_name,
            )
        ]

    if opts.output_nc == 1:
        train_post_transforms = Compose(
            [
                EnsureTypeD(keys=[_pred, _label]),
                ActivationsD(keys=_pred, sigmoid=True),
                # AsDiscreteD(keys=_pred_, threshold_values=True, logit_thresh=0.5),
            ]
        )
    else:
        train_post_transforms = Compose(
            [
                ActivationsD(keys=_pred, softmax=True),
                AsDiscreteD(keys=_pred, argmax=True, to_onehot=None),
                EnsureTypeD(keys=[_pred, _label]),
            ]
        )

    if is_multilabel:
        key_val_metric = Accuracy(
            # output_transform=partial(output_onehot_transform, n_classes=opts.output_nc),
            output_transform=from_engine(
                [_pred, _label],
                onehot_process(opts.output_nc),
            ),
            is_multilabel=is_multilabel,
        )
    else:
        key_val_metric = ROCAUC(
            output_transform=from_engine(
                [_pred, _label], onehot_process(opts.output_nc)
            )
        )

    evaluator = SupervisedEvaluatorEx(
        device=device,
        val_data_loader=test_loader,
        network=net,
        epoch_length=int(opts.n_epoch_len)
        if opts.n_epoch_len > 1.0
        else int(opts.n_epoch_len * len(test_loader)),
        prepare_batch=prepare_batch_fn,
        inferer=SimpleInfererEx(),
        postprocessing=train_post_transforms,
        key_val_metric={val_metric_name: key_val_metric},
        val_handlers=val_handlers,
        amp=opts.amp,
        decollate=decollate,
        custom_keys=cfg.get_keys_dict(),
    )

    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_step_transform = lambda x: evaluator.state.metrics[val_metric_name]
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
            output_transform=from_engine(_loss, first=True),
            name=logger_name,
        ),
        TensorBoardStatsHandler(
            summary_writer=writer,
            tag_name="train_loss",
            output_transform=from_engine(_loss),
        ),
        CheckpointSaverEx(
            save_dir=model_dir / "Checkpoint",
            save_dict={"net": net, "optim": optim},
            save_interval=opts.save_epoch_freq,
            epoch_level=True,
            n_saved=opts.save_n_best,
        ),  #!n_saved=None
        TensorBoardImageHandlerEx(
            summary_writer=writer,
            batch_transform=lambda x: (None, None),
            output_transform=from_engine(_image),
            max_channels=1,
            prefix_name="Train",
        ),
    ]

    trainer = SupervisedTrainerEx(
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
        inferer=SimpleInfererEx(logger_name),
        postprocessing=train_post_transforms,
        key_train_metric={"train_auc": key_val_metric},
        # additional_metrics={"roccurve": add_roc_metric},
        train_handlers=train_handlers,
        amp=opts.amp,
        decollate=decollate,
        custom_keys=cfg.get_keys_dict(),
    )

    if opts.early_stop > 0:
        early_stopper = EarlyStopping(
            patience=opts.early_stop,
            score_function=stopping_fn_from_metric(val_metric_name),
            trainer=trainer,
        )
        evaluator.add_event_handler(
            event_name=Events.EPOCH_COMPLETED, handler=early_stopper
        )

    return trainer


@TEST_ENGINES.register("classification")
def build_classification_test_engine(**kwargs):
    opts = kwargs["opts"]
    test_loader = kwargs["test_loader"]
    net = kwargs["net"]
    device = kwargs["device"]
    logger_name = kwargs.get("logger_name", None)
    is_multilabel = opts.output_nc > 1
    is_supervised = kwargs.get("is_supervised", opts.phase == Phases.TEST_IN)
    multi_input_keys = kwargs.get("multi_input_keys", None)
    multi_output_keys = kwargs.get("multi_output_keys", None)
    output_latent_code = kwargs.get("output_latent_code", False)
    target_latent_layer = kwargs.get("target_latent_layer", None)
    root_dir = output_filename_check(test_loader.dataset)
    decollate = True
    _image = cfg.get_key("image")
    _label = cfg.get_key("label")
    _pred = cfg.get_key("pred")
    _acti = cfg.get_key("forward")

    model_path = (
        opts.model_path[0]
        if isinstance(opts.model_path, (list, tuple))
        else opts.model_path
    )

    if is_supervised:
        prepare_batch_fn = get_prepare_batch_fn(
            opts, _image, _label, multi_input_keys, multi_output_keys
        )
    else:
        prepare_batch_fn = get_unsupervised_prepare_batch_fn(
            opts, _image, multi_input_keys
        )

    def debug(data):
        print("--debug:", type(data), len(data), data)
        return data

    if not is_multilabel:
        saver_post_transform = Compose(
            [
                EnsureTypeD(keys=[_pred, _label]),
                ActivationsD(keys=_pred, sigmoid=True),
                from_engine(_pred)
            ],
            first=False,
        )
        acc_post_transforms = Compose(
            [
                EnsureTypeD(keys=[_pred, _label]),
                ActivationsD(keys=_pred, sigmoid=True),
                AsDiscreteD(keys=_pred, threshold_values=True, logit_thresh=0.5),
                from_engine([_pred, _label], ensure_dim=decollate),
            ],
            first=decollate,
        )
        auc_post_transforms = Compose(
            [
                EnsureTypeD(keys=[_pred, _label]),
                ActivationsD(keys=_pred, sigmoid=True),
                from_engine([_pred, _label], ensure_dim=decollate),
            ],
            first=decollate,
        )
    else:
        saver_post_transform = Compose(
            [
                EnsureTypeD(keys=[_pred, _label]),
                ActivationsD(keys=_pred, softmax=True),
                AsDiscreteD(keys=_pred, argmax=True, to_onehot=None),
                from_engine(_pred)
            ],
            first=False,
        )
        acc_post_transforms = Compose(
            [
                EnsureTypeD(keys=[_pred, _label]),
                ActivationsD(keys=_pred, softmax=True),
                AsDiscreteD(keys=_pred, argmax=True, to_onehot=False),
                from_engine([_pred, _label], onehot_process(opts.output_nc)),
            ]
        )
        auc_post_transforms = Compose(
            [
                EnsureTypeD(keys=[_pred, _label]),
                ActivationsD(keys=_pred, softmax=True),
                AsDiscreteD(keys=_pred, argmax=True, to_onehot=False),
                from_engine([_pred, _label], onehot_process(opts.output_nc)),
            ]
        )

    val_handlers = [
        StatsHandler(name=logger_name),
        CheckpointLoader(load_path=model_path, load_dict={"net": net}),
    ]

    if get_attr_(opts, "save_results", True):
        has_label = opts.phase == Phases.TEST_IN
        val_handlers += [
            ClassificationSaverEx(
                output_dir=opts.out_dir,
                save_labels=has_label,
                batch_transform=from_engine([_image + "_meta_dict", _label])
                if has_label
                else from_engine(_image + "_meta_dict"),
                output_transform=saver_post_transform,
            ),
            ClassificationSaverEx(
                output_dir=opts.out_dir,
                filename="logits.csv",
                batch_transform=from_engine(_image + "_meta_dict"),
                output_transform=from_engine(_pred),
            ),
        ]

    if get_attr_(opts, "save_image", False):
        val_handlers += [
            SegmentationSaver(
                output_dir=opts.out_dir,
                output_postfix=_image,
                data_root_dir=root_dir,
                resample=False,
                mode="bilinear",
                batch_transform=from_engine(_image + "_meta_dict"),
                output_transform=from_engine(_image),
            )
        ]

    if output_latent_code:
        val_handlers += [
            LatentCodeSaver(
                output_dir=opts.out_dir,
                filename="latent",
                data_root_dir=root_dir,
                overwrite=True,
                batch_transform=from_engine(_image + "_meta_dict"),
                output_transform=from_engine(_acti),
                name=logger_name,
                save_to_np=True,
                save_as_onefile=True,
            )
        ]

    if is_supervised:
        evaluator = SupervisedEvaluatorEx(
            device=device,
            val_data_loader=test_loader,
            network=net,
            prepare_batch=prepare_batch_fn,
            inferer=SimpleInfererEx(),  # SlidingWindowClassify(roi_size=opts.crop_size, sw_batch_size=4, overlap=0.3),
            postprocessing=None,  # post_transforms,
            val_handlers=val_handlers,
            key_val_metric={
                "test_acc": Accuracy(
                    output_transform=acc_post_transforms, is_multilabel=is_multilabel
                )
            },
            additional_metrics={
                "test_auc": ROCAUC(output_transform=auc_post_transforms),
                "Prec": Precision(
                    output_transform=acc_post_transforms,
                    average=is_multilabel,
                    is_multilabel=is_multilabel,
                ),
                "Recall": Recall(
                    output_transform=acc_post_transforms,
                    average=is_multilabel,
                    is_multilabel=is_multilabel,
                ),
                "ROC": DrawRocCurve(
                    save_dir=opts.out_dir,
                    output_transform=auc_post_transforms,
                    is_multilabel=is_multilabel,
                ),
            },
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict(),
            output_latent_code=output_latent_code,
            target_latent_layer=target_latent_layer,
        )
    else:
        evaluator = SupervisedEvaluatorEx(
            device=device,
            val_data_loader=test_loader,
            network=net,
            prepare_batch=prepare_batch_fn,
            inferer=SimpleInfererEx(),  # SlidingWindowClassify(roi_size=opts.crop_size, sw_batch_size=4, overlap=0.3),
            postprocessing=None,
            val_handlers=val_handlers,
            key_val_metric=None,
            additional_metrics=None,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict(),
            output_latent_code=output_latent_code,
            target_latent_layer=target_latent_layer,
        )

    return evaluator


@ENSEMBLE_TEST_ENGINES.register("classification")
def build_classification_ensemble_test_engine(**kwargs):
    opts = kwargs["opts"]
    test_loader = kwargs["test_loader"]
    net = kwargs["net"]
    device = kwargs["device"]
    use_best_model = kwargs.get("best_val_model", True)
    model_list = opts.model_path
    is_intra_ensemble = isinstance(model_list, (list, tuple)) and len(model_list) > 0
    logger_name = kwargs.get("logger_name", None)
    logger = logging.getLogger(logger_name)
    is_multilabel = opts.output_nc > 1
    is_supervised = opts.phase == Phases.TEST_IN
    multi_input_keys = kwargs.get("multi_input_keys", None)
    multi_output_keys = kwargs.get("multi_output_keys", None)
    _image = cfg.get_key("image")
    _label = cfg.get_key("label")
    _pred = cfg.get_key("pred")

    cv_folders = [Path(opts.experiment_path) / f"{i}-th" for i in range(opts.n_fold)]
    cv_folders = filter(lambda x: x.is_dir(), cv_folders)
    float_regex = r"=(-?\d+\.\d+).pt"
    int_regex = r"=(\d+).pt"
    if is_intra_ensemble:
        if len(model_list) == 1:
            raise ValueError(
                "Only one model is specified for intra ensemble test, but need more!"
            )
    elif use_best_model:
        model_list = []
        for folder in cv_folders:
            models = list(
                filter(
                    lambda x: re.search(float_regex, x.name),
                    [model for model in (folder / "Models").rglob("*.pt")],
                )  # ? better to usefolder/"Models"/"Best_Models"
            )
            models.sort(key=lambda x: float(re.search(float_regex, x.name).group(1)))
            model_list.append(models[-1])
    else:  # get latest
        for folder in cv_folders:
            models = list(
                filter(
                    lambda x: x.is_file(),
                    [model for model in (folder / "Models" / "Checkpoint").iterdir()],
                )
            )
            try:
                models.sort(key=lambda x: int(re.search(int_regex, x.name).group(1)))
            except AttributeError as e:
                invalid_models = list(
                    filter(lambda x: re.search(int_regex, x.name) is None, models)
                )
                print("invalid models:", invalid_models)
                raise e
            model_list.append(models[-1])

    if len(model_list) != opts.n_fold and not is_intra_ensemble:
        print(
            f"Found {len(model_list)} best models,"
            f"not equal to {opts.n_fold} n_folds.\n"
            f"Use {len(model_list)} best models"
        )
    print(f"Using models: {[m.name for m in model_list]}")

    nets = [
        copy.deepcopy(net),
    ] * len(model_list)
    for net, m in zip(nets, model_list):
        CheckpointLoaderEx(load_path=str(m), load_dict={"net": net}, name=logger_name)(
            None
        )

    pred_keys = [f"{_pred}{i}" for i in range(len(model_list))]

    if is_supervised:
        prepare_batch_fn = get_prepare_batch_fn(
            opts, _image, _label, multi_input_keys, multi_output_keys
        )
    else:
        prepare_batch_fn = get_unsupervised_prepare_batch_fn(
            opts, _image, multi_input_keys
        )

    if opts.output_nc == 1:  # ensemble_type is 'mean':
        if use_best_model:
            w_ = [float(re.search(float_regex, m.name).group(1)) for m in model_list]
        else:
            w_ = None
        post_transforms = MeanEnsembleD(
            keys=pred_keys,
            output_key=_pred,
            # in this particular example, we use validation metrics as weights
            weights=w_,
        )

        acc_post_transforms = Compose(
            [
                ActivationsD(keys=_pred, sigmoid=True),
                AsDiscreteD(keys=_pred, threshold_values=True, logit_thresh=0.5),
                partial(output_onehot_transform, n_classes=opts.output_nc),
            ]
        )
        auc_post_transforms = Compose(
            [
                ActivationsD(keys=_pred, sigmoid=True),
                partial(output_onehot_transform, n_classes=opts.output_nc),
            ]
        )
        ClsSaver_transform = Compose(
            [
                ActivationsD(keys=_pred, sigmoid=True),
                AsDiscreteD(keys=_pred, threshold_values=True, logit_thresh=0.5),
                lambda x: x[_pred].cpu().numpy(),
            ]
        )

    else:  # ensemble_type is 'vote'
        post_transforms = None

        acc_post_transforms = Compose(
            [
                ActivationsD(keys=pred_keys, softmax=True),
                AsDiscreteD(keys=pred_keys, argmax=True, to_onehot=False),
                SqueezeDimD(keys=pred_keys),
                VoteEnsembleD(
                    keys=pred_keys, output_key=_pred, num_classes=opts.output_nc
                ),
                partial(output_onehot_transform, n_classes=opts.output_nc),
            ]
        )
        auc_post_transforms = Compose(
            [
                ActivationsD(keys=pred_keys, softmax=True),
                AsDiscreteD(keys=pred_keys, argmax=True, to_onehot=False),
                SqueezeDimD(keys=pred_keys),
                VoteEnsembleD(
                    keys=pred_keys, output_key=_pred, num_classes=opts.output_nc
                ),
                partial(output_onehot_transform, n_classes=opts.output_nc),
            ]
        )
        ClsSaver_transform = Compose(
            [
                ActivationsD(keys=pred_keys, softmax=True),
                AsDiscreteD(keys=pred_keys, argmax=True, to_onehot=False),
                SqueezeDimD(keys=pred_keys),
                VoteEnsembleD(
                    keys=pred_keys, output_key=_pred, num_classes=opts.output_nc
                ),
            ]
        )

    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        ClassificationSaverEx(
            output_dir=opts.out_dir,
            batch_transform=lambda x: x[_image + "_meta_dict"],
            output_transform=ClsSaver_transform,
        ),
    ]

    if opts.save_image:
        root_dir = output_filename_check(test_loader.dataset)
        val_handlers += [
            SegmentationSaver(
                output_dir=opts.out_dir,
                output_postfix=_image,
                data_root_dir=root_dir,
                resample=False,
                mode="bilinear",
                batch_transform=lambda x: x[_image + "_meta_dict"],
                output_transform=lambda x: x[_image],
            )
        ]

    evaluator = EnsembleEvaluator(
        device=device,
        val_data_loader=test_loader,
        pred_keys=pred_keys,
        networks=nets,
        prepare_batch=prepare_batch_fn,
        inferer=SimpleInfererEx(),
        post_transform=post_transforms,
        val_handlers=val_handlers,
        key_val_metric={
            "test_acc": Accuracy(
                output_transform=acc_post_transforms, is_multilabel=is_multilabel
            )
        },
        additional_metrics={
            "test_auc": ROCAUC(output_transform=auc_post_transforms),
            "Prec": Precision(output_transform=acc_post_transforms),
            "Recall": Recall(output_transform=acc_post_transforms),
            "ROC": DrawRocCurve(
                save_dir=opts.out_dir, output_transform=auc_post_transforms
            ),
        },
    )

    return evaluator
