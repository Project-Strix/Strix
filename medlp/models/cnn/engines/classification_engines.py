import copy
import logging
import re
from types import SimpleNamespace
from functools import partial
from pathlib import Path
from typing import Union, Optional, Sequence, Dict

import torch
from torch.utils.data import DataLoader
from ignite.engine import Events
from ignite.metrics import Accuracy, Precision, Recall
from medlp.configures import config as cfg
from medlp.models.cnn.engines import ENSEMBLE_TEST_ENGINES, TEST_ENGINES, TRAIN_ENGINES
from medlp.models.cnn.engines.engine import MedlpTestEngine, MedlpTrainEngine
from medlp.models.cnn.engines.utils import (
    get_models,
    get_prepare_batch_fn,
    get_unsupervised_prepare_batch_fn,
)
from medlp.models.cnn.utils import onehot_process
from medlp.utilities.enum import Phases
from medlp.utilities.transforms import decollate_transform_adaptor as DTA
from medlp.utilities.utils import output_filename_check, setup_logger, get_attr_
from monai_ex.engines import EnsembleEvaluator, SupervisedEvaluatorEx, SupervisedTrainerEx
from monai_ex.handlers import (
    ROCAUC,
    CheckpointLoaderEx,
    ClassificationSaverEx,
    EarlyStopHandler,
    LatentCodeSaver,
    LrScheduleTensorboardHandler,
    SegmentationSaver,
    StatsHandler,
    ValidationHandler,
)
from monai_ex.handlers import from_engine_ex as from_engine
from monai_ex.handlers import stopping_fn_from_metric
from monai_ex.inferers import SimpleInfererEx
from monai_ex.metrics import DrawRocCurve
from monai_ex.transforms import ActivationsD
from monai_ex.transforms import AsDiscreteExD as AsDiscreteD
from monai_ex.transforms import ComposeEx as Compose
from monai_ex.transforms import EnsureTypeD, GetItemD, MeanEnsembleD, SqueezeDimD, VoteEnsembleD
from monai_ex.utils import ensure_tuple


@TRAIN_ENGINES.register("classification")
class ClassificationTrainEngine(MedlpTrainEngine, SupervisedTrainerEx):
    def __init__(
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
        decollate = False
        _image = cfg.get_key("image")
        _label = cfg.get_key("label")
        _loss = cfg.get_key("loss")

        logger_name = get_attr_(opts, 'logger_name', logger_name)
        self.logger = setup_logger(logger_name)

        key_val_metric = ClassificationTrainEngine.get_metric("val", opts.output_nc, decollate)
        val_metric_name = list(key_val_metric.keys())[0]
        key_train_metric = ClassificationTrainEngine.get_metric("train", opts.output_nc, decollate)
        train_metric_name = list(key_train_metric.keys())[0]

        prepare_batch_fn = get_prepare_batch_fn(opts, _image, _label, multi_input_keys, multi_output_keys)

        val_handlers = MedlpTrainEngine.get_basic_handlers(
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
            tensorboard_image_kwargs=ClassificationTrainEngine.get_tensorboard_image_transform(
                opts.output_nc, decollate
            ),
            dump_tensorboard=True,
            record_nni=opts.nni,
            nni_kwargs={
                "metric_name": val_metric_name,
                "max_epochs": opts.n_epoch,
                "logger_name": logger_name,
            },
        )

        if opts.early_stop > 0:
            val_handlers += [
                EarlyStopHandler(
                    patience=opts.early_stop,
                    score_function=stopping_fn_from_metric(val_metric_name),
                    trainer=self,
                    min_delta=0.0001,
                    epoch_level=True,
                ),
            ]

        evaluator = SupervisedEvaluatorEx(
            device=device,
            val_data_loader=test_loader,
            network=net,
            epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len * len(test_loader)),
            prepare_batch=prepare_batch_fn,
            inferer=SimpleInfererEx(),
            postprocessing=None,
            key_val_metric=key_val_metric,
            val_handlers=val_handlers,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict(),
        )
        evaluator.logger = setup_logger(logger_name)

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_step_transform = lambda x: evaluator.state.metrics[val_metric_name]
        else:
            lr_step_transform = lambda x: ()

        train_handlers = [
            ValidationHandler(validator=evaluator, interval=valid_interval, epoch_level=True),
            LrScheduleTensorboardHandler(
                lr_scheduler=lr_scheduler,
                summary_writer=writer,
                step_transform=lr_step_transform,
            ),
        ]

        train_handlers += MedlpTrainEngine.get_basic_handlers(
            phase="train",
            model_dir=model_dir,
            net=net,
            optimizer=optim,
            tb_summary_writer=writer,
            logger_name=logger_name,
            stats_dicts={"train_loss": from_engine(_loss, first=True)},
            save_checkpoint=True,
            checkpoint_save_interval=opts.save_epoch_freq,
            ckeckpoint_n_saved=opts.save_n_best,
            tensorboard_image_kwargs=ClassificationTrainEngine.get_tensorboard_image_transform(
                opts.output_nc, decollate
            ),
        )

        SupervisedTrainerEx.__init__(
            self,
            device=device,
            max_epochs=opts.n_epoch,
            train_data_loader=train_loader,
            network=net,
            optimizer=optim,
            loss_function=loss,
            epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len * len(train_loader)),
            prepare_batch=prepare_batch_fn,
            inferer=SimpleInfererEx(logger_name),
            postprocessing=None,
            key_train_metric=key_train_metric,
            # additional_metrics={"roccurve": add_roc_metric},
            train_handlers=train_handlers,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict(),
            ensure_dims=True,
        )

    @staticmethod
    def get_metric(phase: str, output_nc: int, decollate: bool, item_index: Optional[int] = None, suffix: str = ''):
        """Return classification engine's metrics.

        Args:
            phase (str): phase name.
            output_nc (int): output channel number.
            decollate (bool): whether use decollate.
            item_index (Optional[int], optional): If network's output and label is tuple, specifiy its index,
                design for multitask compatiblity. Defaults to None.

        Returns:
            dict: {metric_name: metric_fn}
        """
        if output_nc > 1:
            transform = ClassificationTrainEngine.get_acc_post_transform(output_nc, decollate, item_index)
            key_val_metric = Accuracy(output_transform=transform, is_multilabel=True)
            return {f"{phase}_acc_{suffix}": key_val_metric} if suffix else {f"{phase}_acc": key_val_metric}
        else:
            transform = ClassificationTrainEngine.get_auc_post_transform(output_nc, decollate, item_index)
            key_val_metric = ROCAUC(output_transform=transform)
            return {f"{phase}_auc_{suffix}": key_val_metric} if suffix else {f"{phase}_auc": key_val_metric}

    @staticmethod
    def get_acc_post_transform(output_nc, decollate, item_index):
        is_multilabel = output_nc > 1
        _pred = cfg.get_key("pred")
        _label = cfg.get_key("label")

        activate_transform = (
            ActivationsD(keys=_pred, softmax=True) if is_multilabel else ActivationsD(keys=_pred, sigmoid=True)
        )

        discrete_transform = (
            AsDiscreteD(keys=_pred, argmax=True, to_onehot=output_nc, dim=1, keepdim=True)
            if is_multilabel
            else AsDiscreteD(keys=_pred, threshold=0.5)
        )

        onehot_transform = (
            from_engine([_pred, _label], [lambda x: x, onehot_process(output_nc, verbose=True)])
            if is_multilabel
            else from_engine([_pred, _label], ensure_dim=decollate)
        )

        select_item_transform = (
            [DTA(GetItemD(keys=[_pred, _label], index=item_index))] if item_index is not None else []
        )
        transforms = select_item_transform + [
            DTA(EnsureTypeD(keys=[_pred, _label], device="cpu")),
            DTA(activate_transform),
            DTA(discrete_transform),
            onehot_transform,
        ]

        return Compose(transforms, map_items=not decollate)

    @staticmethod
    def get_auc_post_transform(output_nc, decollate, item_index):
        is_multilabel = output_nc > 1
        _pred = cfg.get_key("pred")
        _label = cfg.get_key("label")

        activate_transform = (
            ActivationsD(keys=_pred, softmax=True) if is_multilabel else ActivationsD(keys=_pred, sigmoid=True)
        )

        discrete_transform = AsDiscreteD(keys=_pred, argmax=True, to_onehot=False) if is_multilabel else lambda x: x

        onehot_transform = (
            from_engine([_pred, _label], onehot_process(output_nc, verbose=True))
            if is_multilabel
            else from_engine([_pred, _label], ensure_dim=decollate)
        )

        select_item_transform = (
            [DTA(GetItemD(keys=[_pred, _label], index=item_index))] if item_index is not None else []
        )

        transforms = select_item_transform + [
            DTA(EnsureTypeD(keys=[_pred, _label], device="cpu")),
            DTA(activate_transform),
            DTA(discrete_transform),
            onehot_transform,
        ]

        return Compose(transforms, map_items=not decollate)

    @staticmethod
    def get_tensorboard_image_transform(
        output_nc: int, decollate: bool, item_index: Optional[int] = None, label_key: Optional[str] = None
    ):
        return None


@TEST_ENGINES.register("classification")
class ClassificationTestEngine(MedlpTestEngine, SupervisedEvaluatorEx):
    def __init__(
        self,
        opts,
        test_loader,
        net,
        device,
        logger_name,
        **kwargs,
    ):
        is_supervised = opts.phase == Phases.TEST_IN
        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        output_latent_code = kwargs.get("output_latent_code", False)
        target_latent_layer = kwargs.get("target_latent_layer", None)
        decollate = True
        _image = cfg.get_key("image")
        _label = cfg.get_key("label")
        _pred = cfg.get_key("pred")
        _acti = cfg.get_key("forward")

        model_path = opts.model_path[0] if isinstance(opts.model_path, (list, tuple)) else opts.model_path
        logger_name = get_attr_(opts, 'logger_name', logger_name)
        self.logger = setup_logger(logger_name)

        if is_supervised:
            prepare_batch_fn = get_prepare_batch_fn(opts, _image, _label, multi_input_keys, multi_output_keys)
        else:
            prepare_batch_fn = get_unsupervised_prepare_batch_fn(opts, _image, multi_input_keys)

        key_val_metric = ClassificationTestEngine.get_metric(opts.phase, opts.output_nc, decollate, metric_names="acc")
        key_metric_name = list(key_val_metric.keys())
        additional_val_metrics = ClassificationTestEngine.get_metric(
            opts.phase, opts.output_nc, decollate, output_dir=opts.out_dir, metric_names=["auc", "prec", "recall", "roc"]
        )
        additional_metric_names = list(additional_val_metrics.keys())

        handlers = MedlpTestEngine.get_basic_handlers(
            phase=opts.phase,
            out_dir=opts.out_dir,
            model_path=model_path,
            load_dict={"net": net},
            logger_name=logger_name,
            stats_dicts={"Metrics": lambda x: None},
            save_image=opts.save_image,
            image_resample=False,
            test_loader=test_loader,
            image_batch_transform=from_engine([_image, _image + "_meta_dict"]),
        )

        extra_handlers = ClassificationTestEngine.get_extra_handlers(
            opts=opts, test_loader=test_loader, decollate=decollate, logger_name=logger_name, **kwargs
        )
        if extra_handlers:
            handlers += extra_handlers

        SupervisedEvaluatorEx.__init__(
            self,
            device=device,
            val_data_loader=test_loader,
            network=net,
            prepare_batch=prepare_batch_fn,
            inferer=SimpleInfererEx(),  # SlidingWindowClassify(roi_size=opts.crop_size, sw_batch_size=4, overlap=0.3),
            postprocessing=None,  # post_transforms,
            val_handlers=handlers,
            key_val_metric=key_val_metric if is_supervised else None,
            additional_metrics=additional_val_metrics if is_supervised else None,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict(),
            output_latent_code=output_latent_code,
            target_latent_layer=target_latent_layer,
        )

    @staticmethod
    def get_metric(
        phase: Phases,
        output_nc: int,
        decollate: bool,
        item_index: Optional[int] = None,
        **kwargs,
    ):
        suffix = kwargs.get("suffix", '')
        output_dir = kwargs.get("output_dir", None)
        metric_names = kwargs.get("metric_names", "acc")
        metric_names = ensure_tuple(metric_names)

        is_multilabel = output_nc > 1
        metrics = {}
        for name in metric_names:
            metric_name = ClassificationTestEngine.get_key_metric_name(phase, suffix, metric_name=name)
            if name == "acc":
                transform = ClassificationTrainEngine.get_acc_post_transform(output_nc, decollate, item_index)
                m = {metric_name: Accuracy(output_transform=transform, is_multilabel=is_multilabel)}
            elif name == "auc":
                transform = ClassificationTrainEngine.get_auc_post_transform(output_nc, decollate, item_index)
                m = {metric_name: ROCAUC(output_transform=transform)}
            elif name == "prec":
                transform = ClassificationTrainEngine.get_acc_post_transform(output_nc, decollate, item_index)
                m = {
                    metric_name: Precision(
                        output_transform=transform, average=is_multilabel, is_multilabel=is_multilabel
                    )
                }
            elif name == "recall":
                transform = ClassificationTrainEngine.get_acc_post_transform(output_nc, decollate, item_index)
                m = {
                    metric_name: Recall(
                        output_transform=transform, average=is_multilabel, is_multilabel=is_multilabel
                    )
                }
            elif name == "roc":
                transform = ClassificationTrainEngine.get_auc_post_transform(output_nc, decollate, item_index)
                m = {
                    metric_name: DrawRocCurve(
                        save_dir=output_dir, output_transform=transform, is_multilabel=is_multilabel
                    )
                }
            else:
                raise NotImplementedError(
                    "Currently only support 'acc, auc, prec, recall, roc' as metric, but got {name}.",
                )
            metrics.update(m)
        return metrics

    @staticmethod
    def get_key_metric_name(phase: Phases, suffix: str = '', **kwargs):
        metric_name = kwargs.get("metric_name", "acc")
        if phase == Phases.TEST_EX or phase == Phases.TEST_IN:
            return f"{phase.value}_{metric_name}_{suffix}" if suffix else f"{phase.value}_{metric_name}"
        else:
            return ValueError("Phase not correct for testengine.")

    @staticmethod
    def get_extra_handlers(
        opts: SimpleNamespace,
        test_loader: DataLoader,
        decollate: bool,
        logger_name: str,
        item_index: Optional[int] = None,
        **kwargs: Dict,
    ):
        _image = cfg.get_key("image")
        _label = cfg.get_key("label")
        _acti = cfg.get_key("forward")
        suffix = kwargs.get("suffix", '')
        output_latent_code = kwargs.get("output_latent_code", False)
        root_dir = output_filename_check(test_loader.dataset)
        has_label = opts.phase == Phases.TEST_IN
        output_nc = opts.output_nc if item_index is None else opts.output_nc[item_index]

        extra_handlers = [
            ClassificationSaverEx(
                output_dir=opts.out_dir,
                filename=f"{suffix}_predictions.csv" if suffix else "predictions.csv",
                save_labels=has_label,
                batch_transform=from_engine([_image + "_meta_dict", _label])
                if has_label
                else from_engine(_image + "_meta_dict"),
                output_transform=ClassificationTestEngine.get_cls_saver_post_transform(output_nc, item_index=item_index),
            )
        ]

        if opts.save_prob:
            extra_handlers += [
                ClassificationSaverEx(
                    output_dir=opts.out_dir,
                    filename=f"{suffix}_prob.csv" if suffix else "prob.csv",
                    batch_transform=from_engine(_image + "_meta_dict"),
                    output_transform=ClassificationTestEngine.get_cls_saver_post_transform(
                        output_nc, discrete=False, item_index=item_index
                    ),
                )
            ]

        if output_latent_code:
            extra_handlers += [
                LatentCodeSaver(
                    output_dir=opts.out_dir,
                    filename=f"{suffix}latent",
                    data_root_dir=root_dir,
                    overwrite=True,
                    batch_transform=from_engine(_image + "_meta_dict"),
                    output_transform=from_engine(_acti),
                    name=logger_name,
                    save_to_np=True,
                    save_as_onefile=True,
                )
            ]
        return extra_handlers

    @staticmethod
    def get_cls_saver_post_transform(
        output_nc: int, discrete: bool = True, logit_thresh: float = 0.5, item_index: Optional[int] = None,
    ):
        _pred = cfg.get_key("pred")
        _label = cfg.get_key("label")

        if item_index is not None:
            select_item_transform = [DTA(GetItemD(keys=[_pred, _label], index=item_index))]
        else:
            select_item_transform = []

        if output_nc == 1:
            return Compose(
                select_item_transform + [
                    EnsureTypeD(keys=[_pred, _label]),
                    ActivationsD(keys=_pred, sigmoid=True),
                    AsDiscreteD(keys=_pred, threshold=logit_thresh) if discrete else lambda x: x,
                    from_engine(_pred),
                ],
                first=False,
            )
        else:
            return Compose(
                select_item_transform + [
                    EnsureTypeD(keys=[_pred, _label]),
                    ActivationsD(keys=_pred, softmax=True),
                    AsDiscreteD(keys=_pred, argmax=True, to_onehot=None)  # ? dim=1, keepdim=True
                    if discrete
                    else lambda x: x,
                    from_engine(_pred),
                ],
                first=False,
            )


@ENSEMBLE_TEST_ENGINES.register("classification")
class ClassificationEnsembleTestEngine(MedlpTestEngine, EnsembleEvaluator):
    def __init__(
        self,
        opts: SimpleNamespace,
        test_loader: DataLoader,
        net: Dict,
        device: Union[str, torch.device],
        logger_name: str,
        **kwargs
    ):
        use_best_model = kwargs.get("best_val_model", True)
        model_list = opts.model_path
        logger_name = get_attr_(opts, 'logger_name', logger_name)
        self.logger = setup_logger(logger_name)
        float_regex = r"=(-?\d+\.\d+).pt"
        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        _image = cfg.get_key("image")
        _label = cfg.get_key("label")
        _pred = cfg.get_key("pred")
        decollate = True
        is_intra_ensemble = isinstance(model_list, (list, tuple)) and len(model_list) > 0
        if is_intra_ensemble:
            raise NotImplementedError("Intra ensemble testing not tested yet")

        cv_folders = [Path(opts.experiment_path) / f"{i}-th" for i in range(opts.n_fold)]
        cv_folders = filter(lambda x: x.is_dir(), cv_folders)
        best_models = get_models(cv_folders, "best" if use_best_model else "last")
        model_list = list(filter(lambda x: x is not None and x.is_file(), best_models))

        if len(model_list) != opts.n_fold and not is_intra_ensemble:
            print(
                f"Found {len(model_list)} best models,"
                f"not equal to {opts.n_fold} n_folds.\n"
                f"Use {len(model_list)} best models"
            )
        print(f"Using models: {[m.name for m in model_list]}")

        nets = [copy.deepcopy(net),] * len(model_list)
        for net, m in zip(nets, model_list):
            CheckpointLoaderEx(load_path=str(m), load_dict={"net": net}, name=logger_name)(None)

        pred_keys = [f"{_pred}{i}" for i in range(len(model_list))]
        w_ = [float(re.search(float_regex, m.name).group(1)) for m in best_models] if use_best_model else None

        post_transforms = MeanEnsembleD(
            keys=pred_keys,
            output_key=_pred,
            weights=w_,
        )

        key_val_metric = ClassificationTestEngine.get_metric(opts.phase, opts.output_nc, decollate, metric_names="acc")
        additional_val_metrics = ClassificationTestEngine.get_metric(
            opts.phase, opts.output_nc, decollate, output_dir=opts.out_dir, metric_names=["auc", "prec", "recall", "roc"]
        )

        handlers = MedlpTestEngine.get_basic_handlers(
            phase=opts.phase,
            out_dir=opts.out_dir,
            model_path=best_models,
            load_dict=[{"net": net} for net in nets],
            logger_name=logger_name,
            stats_dicts={"Metrics": lambda x: None},
            save_image=opts.save_image,
            image_resample=opts.resample,
            test_loader=test_loader,
            image_batch_transform=from_engine([_image, _image + "_meta_dict"]),
        )
        extra_handlers = ClassificationTestEngine.get_extra_handlers(
            opts=opts, test_loader=test_loader, decollate=decollate, logger_name=logger_name, **kwargs
        )
        if extra_handlers:
            handlers += extra_handlers

        if opts.phase == Phases.TEST_IN:
            prepare_batch_fn = get_prepare_batch_fn(opts, _image, _label, multi_input_keys, multi_output_keys)
        elif opts.phase == Phases.TEST_EX:
            prepare_batch_fn = get_unsupervised_prepare_batch_fn(opts, _image, multi_input_keys)
            key_val_metric = None
        else:
            raise ValueError(f"Got unexpected phase here {opts.phase}, expect testing.")

        EnsembleEvaluator.__init__(
            self,
            device=device,
            val_data_loader=test_loader,
            networks=nets,
            pred_keys=pred_keys,
            prepare_batch=prepare_batch_fn,
            inferer=SimpleInfererEx(),
            postprocessing=post_transforms,
            key_val_metric=key_val_metric,
            additional_metrics=additional_val_metrics,
            val_handlers=handlers,
            amp=opts.amp,
            decollate=decollate
        )

