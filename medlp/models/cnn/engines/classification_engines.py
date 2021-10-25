from typing import Optional, Callable

import re
import logging
import copy
from pathlib import Path
from functools import partial

import torch
from medlp.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from medlp.models.cnn.engines.utils import (
    output_onehot_transform,
    get_prepare_batch_fn,
    get_unsupervised_prepare_batch_fn,
)
from medlp.utilities.utils import output_filename_check, get_attr_
from medlp.configures import config as cfg
from medlp.models.cnn.engines.engine import MedlpTrainEngine, MedlpTestEngine

from monai_ex.utils import ensure_tuple
from monai_ex.inferers import SimpleInfererEx
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
    Compose,
    ActivationsD,
    AsDiscreteD,
    MeanEnsembleD,
    VoteEnsembleD,
    SqueezeDimD,
)

from monai_ex.handlers import (
    StatsHandler,
    ValidationHandler,
    LrScheduleTensorboardHandler,
    CheckpointSaverEx,
    CheckpointLoader,
    CheckpointLoaderEx,
    SegmentationSaver,
    ClassificationSaverEx,
    ROCAUC,
    stopping_fn_from_metric,
)


@TRAIN_ENGINES.register("classification")
class ClassificationTrainEngine(MedlpTrainEngine):
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

        key_val_metric = ClassificationTrainEngine.get_metric("val", opts.output_nc)
        val_metric_name = list(key_val_metric.keys())[0]

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
            tb_image_batch_transform=lambda x: (None, None),
            tb_image_output_transform=lambda x: x["image"],
            dump_tensorboard=True,
            record_nni=opts.nni,
            nni_kwargs={
                "metric_name": val_metric_name,
                "max_epochs": opts.n_epoch,
                "logger_name": logger_name,
            },
        )

        if opts.output_nc == 1:
            train_post_transforms = Compose(
                [
                    ActivationsD(keys=_pred_, sigmoid=True),
                    # AsDiscreteD(keys=_pred_, threshold_values=True, logit_thresh=0.5),
                ]
            )
        else:
            train_post_transforms = Compose(
                [
                    ActivationsD(keys=_pred_, softmax=True),
                    AsDiscreteD(keys=_pred_, argmax=True, to_onehot=False),
                    SqueezeDimD(keys=_pred_),
                ]
            )

        prepare_batch_fn = get_prepare_batch_fn(
            opts, _image_, _label_, multi_input_keys, multi_output_keys
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
            post_transform=train_post_transforms,
            key_val_metric=key_val_metric,
            val_handlers=val_handlers,
            amp=opts.amp,
            custom_keys=cfg.get_keys_dict(),
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
            tb_image_batch_transform=lambda x: (None, None),
            tb_image_output_transform=lambda x: x["image"],
        )

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
            post_transform=train_post_transforms,
            key_train_metric=ClassificationTrainEngine.get_metric(
                "train", opts.output_nc
            ),
            # additional_metrics={"roccurve": add_roc_metric},
            train_handlers=train_handlers,
            amp=opts.amp,
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

    @staticmethod
    def get_metric(phase: str, output_nc: int):
        if output_nc > 1:
            key_val_metric = Accuracy(
                output_transform=partial(output_onehot_transform, n_classes=output_nc),
                is_multilabel=is_multilabel,
            )
            return {phase + "_acc": key_val_metric}
        else:
            key_val_metric = ROCAUC(
                output_transform=partial(output_onehot_transform, n_classes=output_nc)
            )
            return {phase + "_auc": key_val_metric}


@TEST_ENGINES.register("classification")
class ClassificationTestEngine(MedlpTestEngine):
    def __new__(
        self,
        opts,
        test_loader,
        net,
        device,
        logger_name,
        **kwargs,
    ):
        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        is_multilabel = opts.output_nc > 1
        is_supervised = opts.phase == "test"
        _image_ = cfg.get_key("image")
        _label_ = cfg.get_key("label")
        _pred_ = cfg.get_key("pred")
        model_path = (
            opts.model_path[0]
            if isinstance(opts.model_path, (list, tuple))
            else opts.model_path
        )

        if is_supervised:
            prepare_batch_fn = get_prepare_batch_fn(
                opts, _image_, _label_, multi_input_keys, multi_output_keys
            )
        else:
            prepare_batch_fn = get_unsupervised_prepare_batch_fn(
                opts, _image_, multi_input_keys
            )

        if is_multilabel:
            post_transform = Compose(
                [
                    ActivationsD(keys=_pred_, softmax=True),
                    AsDiscreteD(keys=_pred_, argmax=True, to_onehot=False),
                    SqueezeDimD(keys=_pred_),
                    lambda x: x[_pred_].cpu().numpy(),
                ]
            )
        else:
            post_transform = Compose(
                [
                    ActivationsD(keys=_pred_, sigmoid=True),
                    lambda x: x[_pred_].cpu().numpy(),
                ]
            )

        key_val_metric = ClassificationTestEngine.get_metric("acc", opts.output_nc)
        key_metric_name = list(key_val_metric.keys())[0]

        val_handlers = MedlpTestEngine.get_extra_handlers(
            phase=opts.phase,
            out_dir=opts.out_dir,
            model_path=model_path,
            net=net,
            logger_name=logger_name,
            stats_dicts={key_metric_name: lambda x: None},
            save_image=opts.save_image,
            image_resample=False,
            test_loader=test_loader,
            image_batch_transform=lambda x: x[_image_ + "_meta_dict"],
            image_output_transform=lambda x: x[_image_],
        )

        if get_attr_(opts, "save_results", True):
            val_handlers += [
                ClassificationSaverEx(
                    output_dir=opts.out_dir,
                    save_labels=opts.phase == "test",
                    batch_transform=lambda x: (x[_image_ + "_meta_dict"], x[_label_])
                    if opts.phase == "test"
                    else x[_image_ + "_meta_dict"],
                    output_transform=post_transform,
                ),
                ClassificationSaverEx(
                    output_dir=opts.out_dir,
                    filename="logits.csv",
                    batch_transform=lambda x: x[_image_ + "_meta_dict"],
                    output_transform=lambda x: x[_pred_].cpu().numpy(),
                ),
            ]

        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=test_loader,
            network=net,
            prepare_batch=prepare_batch_fn,
            inferer=SimpleInfererEx(),  # SlidingWindowClassify(roi_size=opts.crop_size, sw_batch_size=4, overlap=0.3),
            post_transform=None,  # post_transforms,
            val_handlers=val_handlers,
            key_val_metric=key_val_metric if is_supervised else None,
            additional_metrics=ClassificationTestEngine.get_metric(
                ["auc", "prec", "recall", "roc"], opts.output_nc, opts.out_dir
            )
            if is_supervised
            else None,
            amp=opts.amp,
        )

        return evaluator

    @staticmethod
    def get_metric(metric_names, output_nc, output_dir=None):
        metric_names = ensure_tuple(metric_names)
        is_multilabel = output_nc > 1
        metrics = {}
        for name in metric_names:
            if name == "acc":
                transform = ClassificationTestEngine.get_acc_post_transform(output_nc)
                m = {
                    f"test_{name}": Accuracy(
                        output_transform=transform, is_multilabel=is_multilabel
                    )
                }
            elif name == "auc":
                transform = ClassificationTestEngine.get_auc_post_transform(output_nc)
                m = {
                    f"test_{name}": ROCAUC(output_transform=transform),
                }
            elif name == "prec":
                transform = ClassificationTestEngine.get_acc_post_transform(output_nc)
                m = {
                    f"test_{name}": Precision(
                        output_transform=transform,
                        average=is_multilabel,
                        is_multilabel=is_multilabel,
                    )
                }
            elif name == "recall":
                transform = ClassificationTestEngine.get_acc_post_transform(output_nc)
                m = {
                    f"test_{name}": Recall(
                        output_transform=transform,
                        average=is_multilabel,
                        is_multilabel=is_multilabel,
                    ),
                }
            elif name == "roc":
                transform = ClassificationTestEngine.get_auc_post_transform(output_nc)
                m = {
                    f"test_{name}": DrawRocCurve(
                        save_dir=out_dir,
                        output_transform=transform,
                        is_multilabel=is_multilabel,
                    ),
                }
            else:
                raise NotImplementedError(
                    "Currently only support 'acc, auc, prec, recall, roc' as metric",
                    f"but got {name}.",
                )
            metrics.update(m)
            return metrics

    @staticmethod
    def get_acc_post_transform(output_nc):
        is_multilabel = output_nc > 1
        _pred_ = cfg.get_key("pred")

        activation_tsf = (
            ActivationsD(keys=_pred_, softmax=True)
            if is_multilabel
            else ActivationsD(keys=_pred_, sigmoid=True)
        )

        discrete_tsf = (
            AsDiscreteD(keys=_pred_, argmax=True, to_onehot=False)
            if is_multilabel
            else AsDiscreteD(keys=_pred_, threshold_values=True, logit_thresh=0.5)
        )

        squeeze_tsf = SqueezeDimD(keys=_pred_) if is_multilabel else lambda x: x

        return Compose(
            [
                activation_tsf,
                discrete_tsf,
                squeeze_tsf,
                partial(output_onehot_transform, n_classes=output_nc),
            ]
        )

    @staticmethod
    def get_auc_post_transform(output_nc):
        is_multilabel = output_nc > 1
        _pred_ = cfg.get_key("pred")

        activation_tsf = (
            ActivationsD(keys=_pred_, softmax=True)
            if is_multilabel
            else ActivationsD(keys=_pred_, sigmoid=True)
        )

        discrete_tsf = (
            AsDiscreteD(keys=_pred_, argmax=True, to_onehot=False)
            if is_multilabel
            else lambda x: x
        )

        squeeze_tsf = SqueezeDimD(keys=_pred_) if is_multilabel else lambda x: x

        return Compose(
            [
                activation_tsf,
                discrete_tsf,
                squeeze_tsf,
                partial(output_onehot_transform, n_classes=output_nc),
            ]
        )


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
    is_supervised = opts.phase == "test"
    multi_input_keys = kwargs.get("multi_input_keys", None)
    multi_output_keys = kwargs.get("multi_output_keys", None)
    _image_ = cfg.get_key("image")
    _label_ = cfg.get_key("label")
    _pred_ = cfg.get_key("pred")

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

    pred_keys = [f"{_pred_}{i}" for i in range(len(model_list))]

    if is_supervised:
        prepare_batch_fn = get_prepare_batch_fn(
            opts, _image_, _label_, multi_input_keys, multi_output_keys
        )
    else:
        prepare_batch_fn = get_unsupervised_prepare_batch_fn(
            opts, _image_, multi_input_keys
        )

    if opts.output_nc == 1:  # ensemble_type is 'mean':
        if use_best_model:
            w_ = [float(re.search(float_regex, m.name).group(1)) for m in model_list]
        else:
            w_ = None
        post_transforms = MeanEnsembleD(
            keys=pred_keys,
            output_key=_pred_,
            # in this particular example, we use validation metrics as weights
            weights=w_,
        )

        acc_post_transforms = Compose(
            [
                ActivationsD(keys=_pred_, sigmoid=True),
                AsDiscreteD(keys=_pred_, threshold_values=True, logit_thresh=0.5),
                partial(output_onehot_transform, n_classes=opts.output_nc),
            ]
        )
        auc_post_transforms = Compose(
            [
                ActivationsD(keys=_pred_, sigmoid=True),
                partial(output_onehot_transform, n_classes=opts.output_nc),
            ]
        )
        ClsSaver_transform = Compose(
            [
                ActivationsD(keys=_pred_, sigmoid=True),
                AsDiscreteD(keys=_pred_, threshold_values=True, logit_thresh=0.5),
                lambda x: x[_pred_].cpu().numpy(),
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
                    keys=pred_keys, output_key=_pred_, num_classes=opts.output_nc
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
                    keys=pred_keys, output_key=_pred_, num_classes=opts.output_nc
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
                    keys=pred_keys, output_key=_pred_, num_classes=opts.output_nc
                ),
            ]
        )

    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        ClassificationSaverEx(
            output_dir=opts.out_dir,
            batch_transform=lambda x: x[_image_ + "_meta_dict"],
            output_transform=ClsSaver_transform,
        ),
    ]

    if opts.save_image:
        root_dir = output_filename_check(test_loader.dataset)
        val_handlers += [
            SegmentationSaver(
                output_dir=opts.out_dir,
                output_postfix=_image_,
                data_root_dir=root_dir,
                resample=False,
                mode="bilinear",
                batch_transform=lambda x: x[_image_ + "_meta_dict"],
                output_transform=lambda x: x[_image_],
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
