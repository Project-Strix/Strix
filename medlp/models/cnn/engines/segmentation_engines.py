from typing import Optional, Callable
import re
import logging
import copy
from pathlib import Path
from functools import partial

import torch
from medlp.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from medlp.models.cnn.engines.utils import (
    get_prepare_batch_fn,
    get_unsupervised_prepare_batch_fn,
    get_dice_metric_transform_fn,
)
from medlp.utilities.utils import is_avaible_size, output_filename_check, get_attr_
from medlp.configures import config as cfg
from medlp.models.cnn.engines.engine import MedlpTrainEngine, MedlpTestEngine
from medlp.models.cnn.engines.utils import get_models

from monai_ex.inferers import (
    SimpleInferer,
    SlidingWindowInferer,
    SlidingWindowInferer2Dfor3D,
)
from monai_ex.networks import one_hot
from ignite.engine import Events
from ignite.handlers import EarlyStopping

from monai_ex.engines import SupervisedTrainer, SupervisedEvaluator, EnsembleEvaluator

from monai_ex.transforms import Compose, ActivationsD, AsDiscreteD, MeanEnsembleD
from monai_ex.handlers import (
    StatsHandler,
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
class SegmentationTrainEngine(MedlpTrainEngine):
    def __new__(
        self,
        opts,
        train_loader,
        test_loader,
        net,
        loss,
        optim,
        lr_scheduler,
        writer,
        device,
        model_dir,
        logger_name,
        **kwargs,
    ):
        valid_interval = kwargs.get("valid_interval", 1)
        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        _image_ = cfg.get_key("image")
        _label_ = cfg.get_key("label")
        _pred_ = cfg.get_key("pred")
        _loss_ = cfg.get_key("loss")

        val_metric = SegmentationTrainEngine.get_key_metric(
            phase="val", output_nc=opts.output_nc, criterion=opts.criterion
        )
        train_metric = SegmentationTrainEngine.get_key_metric(
            phase="train", output_nc=opts.output_nc, criterion=opts.criterion
        )
        val_metric_name = list(val_metric.keys())[0]

        val_handlers = MedlpTrainEngine.get_extra_handlers(
            phase="val",
            model_dir=model_dir,
            net=net,
            optimizer=optim,
            tb_summary_writer=writer,
            logger_name=logger_name,
            stats_dicts={val_metric_name: lambda x: None},
            save_bestmodel=True,
            model_file_prefix=val_metric_name,
            bestmodel_n_saved=opts.save_n_best,
            tb_show_image=True,
            tb_image_batch_transform=lambda x: (x["image"], x["label"]),
            tb_image_output_transform=lambda x: x["pred"],
            dump_tensorboard=True,
            record_nni=opts.nni,
            nni_kwargs={
                "metric_name": val_metric_name,
                "max_epochs": opts.n_epoch,
                "logger_name": logger_name,
            },
        )

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
                        keys=_pred_,
                        to_onehot=True,
                        argmax=True,
                        n_classes=opts.output_nc,
                    ),
                    # KeepLargestConnectedComponentD(keys=_pred_, applied_labels=[1], independent=False),
                ]
            )

        prepare_batch_fn = get_prepare_batch_fn(
            opts, _image_, _label_, multi_input_keys, multi_output_keys
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
            key_val_metric=val_metric,
            val_handlers=val_handlers,
            amp=opts.amp,
        )

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_step_transform = lambda x: evaluator.state.metrics[val_metric_name]
        else:
            lr_step_transform = lambda x: ()

        train_handlers = [
            ValidationHandler(
                validator=evaluator, interval=valid_interval, epoch_level=True
            ),
            LrScheduleTensorboardHandler(
                lr_scheduler=lr_scheduler,
                summary_writer=writer,
                step_transform=lr_step_transform,
            ),
        ]
        train_handlers += MedlpTrainEngine.get_extra_handlers(
            phase="train",
            model_dir=model_dir,
            net=net,
            optimizer=optim,
            tb_summary_writer=writer,
            logger_name=logger_name,
            stats_dicts={"train_loss": lambda x: x[_loss_]},
            save_checkpoint=True,
            checkpoint_save_interval=opts.save_epoch_freq,
            ckeckpoint_n_saved=opts.save_n_best,
            tb_show_image=True,
            tb_image_batch_transform=lambda x: (x["image"], x["label"]),
            tb_image_output_transform=lambda x: x["pred"],
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
            key_train_metric=train_metric,
            train_handlers=train_handlers,
            amp=opts.amp,
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

    @staticmethod
    def get_key_metric(phase: str, output_nc: int, criterion: str):
        dice_metric_transform_fn = get_dice_metric_transform_fn(
            output_nc,
            criterion,
            pred_key=cfg.get_key("pred"),
            label_key=cfg.get_key("label"),
        )

        key_metric = MeanDice(
            include_background=False, output_transform=dice_metric_transform_fn
        )
        return {phase + "_mean_dice": key_metric}


@TEST_ENGINES.register("segmentation")
class SegmentationTestEngine(MedlpTestEngine):
    def __new__(
        self,
        opts,
        test_loader,
        net,
        device,
        logger_name,
        **kwargs,
    ):
        crop_size = get_attr_(opts, "crop_size", None)
        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        n_batch = opts.n_batch
        resample = opts.resample
        use_slidingwindow = opts.slidingwindow
        is_multilabel = opts.output_nc > 1
        is_supervised = opts.phase == "test"
        _image_ = cfg.get_key("image")
        _pred_ = cfg.get_key("pred")
        _label_ = cfg.get_key("label")
        model_path = (
            opts.model_path[0]
            if isinstance(opts.model_path, (list, tuple))
            else opts.model_path
        )

        key_val_metric = SegmentationTestEngine.get_key_metric(
            opts.phase, opts.output_nc
        )
        val_metric_name = list(key_val_metric.keys())[0]

        if use_slidingwindow:
            print("---Use slidingwindow infer!---")
            print("patch size:", crop_size)
        else:
            print("---Use simple infer!---")

        saver_post_transform = (
            Compose(
                [
                    ActivationsD(keys=_pred_, sigmoid=True),
                    AsDiscreteD(keys=_pred_, threshold_values=True, logit_thresh=0.5),
                    lambda output: output[_pred_],
                ]
            )
            if opts.output_nc == 1
            else Compose(
                [
                    ActivationsD(keys=_pred_, softmax=True),
                    AsDiscreteD(
                        keys=_pred_,
                        argmax=True,
                        to_onehot=False,
                        n_classes=opts.output_nc,
                    ),
                    lambda output: output[_pred_],
                ]
            )
        )

        val_handlers = [
            SegmentationSaver(
                output_dir=opts.out_dir,
                output_ext=".nii.gz",
                resample=resample,
                data_root_dir=output_filename_check(test_loader.dataset),
                batch_transform=lambda x: x[_image_ + "_meta_dict"],
                output_transform=saver_post_transform,
            ),
        ]

        val_handlers += MedlpTestEngine.get_extra_handlers(
            phase=opts.phase,
            out_dir=opts.out_dir,
            model_path=model_path,
            net=net,
            logger_name=logger_name,
            stats_dicts={val_metric_name: lambda x: None},
            save_image=opts.save_image,
            image_resample=resample,
            test_loader=test_loader,
            image_batch_transform=lambda x: x[_image_ + "_meta_dict"],
            image_output_transform=lambda output: output[_image_],
        )

        if is_supervised:
            prepare_batch_fn = get_prepare_batch_fn(
                opts, _image_, _label_, multi_input_keys, multi_output_keys
            )
        else:
            prepare_batch_fn = get_unsupervised_prepare_batch_fn(
                opts, _image_, multi_input_keys
            )

        if use_slidingwindow:
            if opts.tensor_dim == "2D":
                inferer = SlidingWindowInferer2Dfor3D(
                    roi_size=crop_size, sw_batch_size=n_batch, overlap=0.6
                )
            elif opts.tensor_dim == "3D":
                inferer = SlidingWindowInferer(
                    roi_size=crop_size, sw_batch_size=n_batch, overlap=0.5
                )
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

    @staticmethod
    def get_key_metric(phase: str, output_nc: int):
        _pred_ = cfg.get_key("pred")

        def onehot_transform(output, n_classes=3, dim=1):
            if n_classes == 1:
                return output[_pred_], output[_label_]
            y_pred, y = output[_pred_], output[_label_]
            return one_hot(y_pred, n_classes, dim=dim), one_hot(y, n_classes, dim=dim)

        post_transforms = (
            Compose(
                [
                    ActivationsD(keys=_pred_, sigmoid=True),
                    AsDiscreteD(keys=_pred_, threshold_values=True, logit_thresh=0.5),
                    partial(onehot_transform, n_classes=output_nc),
                ]
            )
            if output_nc == 1
            else Compose(
                [
                    ActivationsD(keys=_pred_, softmax=True),
                    AsDiscreteD(
                        keys=_pred_,
                        argmax=True,
                        to_onehot=False,
                        n_classes=output_nc,
                    ),
                    partial(onehot_transform, n_classes=output_nc),
                ]
            )
        )

        if phase == "test_wo_label":
            return {"val_mean_dice": None}
        elif phase == "test":
            return {
                "val_mean_dice": MeanDice(
                    include_background=False, output_transform=post_transforms
                )
            }


@ENSEMBLE_TEST_ENGINES.register("segmentation")
def build_segmentation_ensemble_test_engine(**kwargs):
    raise NotImplementedError("Not finished!")

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
    use_slidingwindow = is_avaible_size(opts.crop_size)
    float_regex = r"=(-?\d+\.\d+).pt"
    if is_intra_ensemble:
        raise NotImplementedError()

    cv_folders = [Path(opts.experiment_path) / f"{i}-th" for i in range(opts.n_fold)]
    cv_folders = filter(lambda x: x.is_dir(), cv_folders)
    best_models = get_models(cv_folders, "best" if use_best_model else "last")
    best_models = list(filter(lambda x: x is not None and x.is_file(), best_models))

    if len(best_models) != opts.n_fold:
        print(
            f"Found {len(best_models)} best models,"
            f"not equal to {opts.n_fold} n_folds.\n"
            f"Use {len(best_models)} best models"
        )
    print(f"Using models: {[m.name for m in best_models]}")

    nets = [
        copy.deepcopy(net),
    ] * len(best_models)
    for net, m in zip(nets, best_models):
        CheckpointLoader(load_path=str(m), load_dict={"net": net}, name=logger_name)(
            None
        )

    pred_keys = [f"pred{i}" for i in range(len(best_models))]

    if use_best_model:
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
        key_val_metric={
            "test_acc": Accuracy(
                output_transform=acc_post_transforms, is_multilabel=is_multilabel
            )
        },
        additional_metrics={
            "test_auc": ROCAUC(output_transform=auc_post_transforms),
            "Prec": Precision(output_transform=acc_post_transforms),
            "Recall": Recall(output_transform=acc_post_transforms),
        },
    )

    return evaluator
