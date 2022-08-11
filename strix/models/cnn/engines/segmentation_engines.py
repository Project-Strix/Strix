from types import SimpleNamespace
from typing import Optional, Union, Sequence, Dict
import re
import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader 
from strix.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from strix.models.cnn.engines.utils import (
    get_models,
    get_prepare_batch_fn,
    get_unsupervised_prepare_batch_fn,
    get_dice_metric_transform_fn,
)
from strix.utilities.utils import setup_logger, output_filename_check, get_attr_
from strix.utilities.enum import Phases
from strix.utilities.transforms import decollate_transform_adaptor as DTA
from strix.configures import config as cfg
from strix.models.cnn.engines.engine import StrixTrainEngine, StrixTestEngine

from monai_ex.inferers import SimpleInfererEx as SimpleInferer, SlidingWindowInferer

from monai_ex.engines import SupervisedTrainerEx, SupervisedEvaluatorEx, EnsembleEvaluator

from monai_ex.transforms import (
    ComposeEx as Compose,
    ActivationsD,
    AsDiscreteExD as AsDiscreteD,
    MeanEnsembleD,
    GetItemD,
)
from monai_ex.handlers import (
    ValidationHandler,
    LrScheduleTensorboardHandler,
    SegmentationSaver,
    MeanDice,
    stopping_fn_from_metric,
    from_engine_ex as from_engine,
    EarlyStopHandler,
    ImageBatchSaver,
)


@TRAIN_ENGINES.register("segmentation")
class SegmentationTrainEngine(StrixTrainEngine, SupervisedTrainerEx):
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
        _image = cfg.get_key("image")
        _label = cfg.get_key("label")
        _loss = cfg.get_key("loss")
        decollate = False
        logger_name = get_attr_(opts, 'logger_name', logger_name)

        val_metric = SegmentationTrainEngine.get_metric(Phases.VALID, output_nc=opts.output_nc, decollate=decollate)
        train_metric = SegmentationTrainEngine.get_metric(Phases.TRAIN, output_nc=opts.output_nc, decollate=decollate)
        val_metric_name = list(val_metric.keys())[0]

        val_handlers = StrixTrainEngine.get_basic_handlers(
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
            tensorboard_image_kwargs=SegmentationTrainEngine.get_tensorboard_image_transform(
                output_nc=opts.output_nc, decollate=decollate, interval=opts.tb_dump_img_interval // valid_interval
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

        prepare_batch_fn = get_prepare_batch_fn(opts, _image, _label, multi_input_keys, multi_output_keys)

        evaluator = SupervisedEvaluatorEx(
            device=device,
            val_data_loader=test_loader,
            network=net,
            epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len * len(test_loader)),
            prepare_batch=prepare_batch_fn,
            inferer=SimpleInferer(),
            postprocessing=None,
            key_val_metric=val_metric,
            val_handlers=val_handlers,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict()
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
                name=logger_name,
                step_transform=lr_step_transform,
            ),
        ]
        train_handlers += StrixTrainEngine.get_basic_handlers(
            phase="train",
            model_dir=model_dir,
            net=net,
            optimizer=optim,
            tb_summary_writer=writer,
            logger_name=logger_name,
            stats_dicts={"train_loss": from_engine(_loss, first=True)},
            save_checkpoint=True,
            checkpoint_save_interval=opts.save_epoch_freq,
            ckeckpoint_n_saved=1,
            tensorboard_image_kwargs=SegmentationTrainEngine.get_tensorboard_image_transform(
                output_nc=opts.output_nc, decollate=decollate, interval=opts.tb_dump_img_interval
            ),
            graph_batch_transform=prepare_batch_fn if opts.visualize else None,
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
            inferer=SimpleInferer(),
            postprocessing=None,
            key_train_metric=train_metric,
            train_handlers=train_handlers,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict(),
            ensure_dims=True,
        )
        self.logger = setup_logger(logger_name)

    @staticmethod
    def get_metric(phase: Phases, output_nc: int, decollate: bool, item_index: Optional[int] = None, suffix: str = ''):
        transform = SegmentationTrainEngine.get_dice_post_transform(output_nc, decollate, item_index)
        key_metric = MeanDice(include_background=False, output_transform=transform)
        return {f"{phase.value}_mean_dice_{suffix}": key_metric} if suffix else {f"{phase.value}_mean_dice": key_metric}

    @staticmethod
    def get_dice_post_transform(output_nc: int, decollate: bool, item_index: Optional[int] = None):
        _pred = cfg.get_key("pred")
        _label = cfg.get_key("label")

        select_item_transform = (
            [DTA(GetItemD(keys=[_pred, _label], index=item_index))] if item_index is not None else []
        )

        if output_nc == 1:
            return Compose(
                select_item_transform
                + [
                    #! DTA is a tmp solution for decollate
                    DTA(ActivationsD(keys=_pred, sigmoid=True)),
                    DTA(AsDiscreteD(keys=_pred, threshold=0.5)),
                    get_dice_metric_transform_fn(
                        output_nc,
                        pred_key=_pred,
                        label_key=_label,
                        decollate=decollate,
                    ),
                ],
                map_items=not decollate,
            )
        else:
            return Compose(
                select_item_transform
                + [
                    DTA(ActivationsD(keys=_pred, softmax=True)),
                    DTA(AsDiscreteD(keys=_pred, argmax=True, to_onehot=output_nc))
                    if decollate
                    else AsDiscreteD(keys=_pred, argmax=True, to_onehot=output_nc, dim=1, keepdim=True),
                    get_dice_metric_transform_fn(
                        output_nc,
                        pred_key=_pred,
                        label_key=_label,
                        decollate=decollate,
                    ),
                ],
                map_items=not decollate,
            )

    @staticmethod
    def get_tensorboard_image_transform(
        output_nc: int, decollate: bool, interval: int, item_index: Optional[int] = None, label_key: Optional[str] = None
    ):
        _image = cfg.get_key("image")
        _label = label_key if label_key else cfg.get_key("label")
        _pred = cfg.get_key("pred")
        if output_nc == 1:
            post_transform = Compose(
                [
                    DTA(GetItemD(keys=_pred, index=item_index)) if item_index is not None else lambda x: x,
                    DTA(ActivationsD(keys=_pred, sigmoid=True)),
                    DTA(AsDiscreteD(keys=_pred, threshold=0.5)),
                    from_engine(_pred)
                ],
                map_items=not decollate,
            )
        else:
            post_transform = Compose(
                [
                    DTA(GetItemD(keys=_pred, index=item_index)) if item_index is not None else lambda x: x,
                    DTA(ActivationsD(keys=_pred, softmax=True)),
                    DTA(AsDiscreteD(keys=_pred, argmax=True, to_onehot=output_nc))
                    if decollate 
                    else AsDiscreteD(keys=_pred, argmax=True, to_onehot=output_nc, dim=1, keepdim=True),
                    from_engine(_pred)
                ],
                map_items=not decollate,
            )

        return {
            "batch_transform": from_engine([_image, _label]),
            "output_transform": post_transform,
            "max_channels": 6,
            "interval": interval,
        }


@TEST_ENGINES.register("segmentation")
class SegmentationTestEngine(StrixTestEngine, SupervisedEvaluatorEx):
    def __init__(
        self,
        opts,
        test_loader,
        net,
        device,
        logger_name,
        **kwargs,
    ):
        crop_size = get_attr_(opts, "crop_size", None)
        use_slidingwindow = get_attr_(opts, "slidingwindow", None)
        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        _image = cfg.get_key("image")
        _label = cfg.get_key("label")
        decollate = True
        logger_name = get_attr_(opts, 'logger_name', logger_name)
        self.logger = setup_logger(logger_name)

        if use_slidingwindow:
            print("---Use slidingwindow infer!---","\nPatch size:", crop_size)
        else:
            print("---Use simple infer!---")
        
        key_val_metric = SegmentationTestEngine.get_metric(opts.phase, opts.output_nc, decollate)
        metric_name = SegmentationTestEngine.get_key_metric_name(opts.phase)

        model_path = opts.model_path[0] if isinstance(opts.model_path, (list, tuple)) else opts.model_path
        handlers = StrixTestEngine.get_basic_handlers(
            phase=opts.phase,
            out_dir=opts.out_dir,
            model_path=model_path,
            load_dict={"net": net},
            logger_name=logger_name,
            stats_dicts={metric_name: lambda x: None},
            save_image=opts.save_image,
            image_resample=opts.resample,
            test_loader=test_loader,
            image_batch_transform=from_engine([_image, _image + "_meta_dict"]),
        )
        extra_handlers = SegmentationTestEngine.get_extra_handlers(
            opts=opts, test_loader=test_loader, decollate=decollate, logger_name=logger_name, **kwargs
        )
        if extra_handlers:
            handlers += extra_handlers

        if opts.phase == Phases.TEST_EX:
            prepare_batch_fn = get_unsupervised_prepare_batch_fn(opts, _image, multi_input_keys)
            key_val_metric = None
        elif opts.phase == Phases.TEST_IN:
            prepare_batch_fn = get_prepare_batch_fn(opts, _image, _label, multi_input_keys, multi_output_keys)

        if use_slidingwindow:
            inferer = SlidingWindowInferer(roi_size=crop_size, sw_batch_size=opts.n_batch, overlap=0.5)
        else:
            inferer = SimpleInferer()

        SupervisedEvaluatorEx.__init__(
            self,
            device=device,
            val_data_loader=test_loader,
            network=net,
            prepare_batch=prepare_batch_fn,
            inferer=inferer,
            postprocessing=None,
            key_val_metric=key_val_metric,
            val_handlers=handlers,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict()
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

        transform = SegmentationTrainEngine.get_dice_post_transform(output_nc, decollate, item_index)
        if phase == Phases.TEST_EX:
            return {SegmentationTestEngine.get_key_metric_name(phase, suffix): None}
        elif phase == Phases.TEST_IN:
            return {
                SegmentationTestEngine.get_key_metric_name(phase, suffix): 
                MeanDice(include_background=False, output_transform=transform)
            }

    @staticmethod
    def get_key_metric_name(phase: Phases, suffix: str = ''):
        if phase == Phases.TEST_EX or phase == Phases.TEST_IN:
            return f"{phase.value}_mean_dice_{suffix}" if suffix else f"{phase.value}_mean_dice"
        else:
            raise ValueError("Phase not correct for testengine.")

    @staticmethod
    def get_extra_handlers(
        opts: SimpleNamespace,
        test_loader: DataLoader,
        decollate: bool,
        logger_name: str,
        item_index: Optional[int] = None,
        **kwargs: Dict,
    ) -> Sequence:
        _image = cfg.get_key("image")
        suffix = kwargs.get("suffix", '')
        output_nc = opts.output_nc if item_index is None else opts.output_nc[item_index]

        data_root_dir = output_filename_check(test_loader.dataset, meta_key=_image+"_meta_dict")
        extra_handlers = [
            SegmentationSaver(
                output_dir=opts.out_dir,
                output_postfix=f"{suffix}_seg",
                output_ext=".nii.gz",
                resample=opts.resample,
                data_root_dir=data_root_dir,
                batch_transform=from_engine(_image + "_meta_dict"), 
                output_transform=SegmentationTestEngine.get_seg_saver_post_transform(
                    output_nc, decollate=decollate, item_index=item_index),
            ),
        ]

        if opts.save_prob:
            extra_handlers += [
                SegmentationSaver(
                    output_dir=opts.out_dir,
                    output_postfix=f"{suffix}_prob",
                    output_ext=".nii.gz",
                    resample=opts.resample,
                    data_root_dir=data_root_dir,
                    batch_transform=from_engine(_image + "_meta_dict"),
                    output_transform=SegmentationTestEngine.get_seg_saver_post_transform(
                        output_nc, decollate, discrete=False, item_index=item_index
                    ),
                ),
            ]
        
        if opts.save_label:
            multi_output_keys = kwargs.get("multi_output_keys", None)
            if multi_output_keys and item_index is not None:
                _label = multi_output_keys[item_index]
            else:
                _label = cfg.get_key("label")

            extra_handlers += [
                ImageBatchSaver(
                    output_dir=opts.out_dir,
                    output_ext=".nii.gz",
                    output_postfix=f"{suffix}_label",
                    data_root_dir=output_filename_check(test_loader.dataset, meta_key=_label+"_meta_dict"),
                    resample=False,  # opts.resample,
                    mode="bilinear",
                    image_batch_transform=from_engine([_label, _label+"_meta_dict"]),
                )                
            ]

        return extra_handlers

    @staticmethod
    def get_seg_saver_post_transform(
        output_nc: int,
        decollate: bool,
        discrete: bool = True,
        logit_thresh: float = 0.5,
        item_index: Optional[int] = None,
    ):
        _pred = cfg.get_key("pred")
        select_item_transform = [DTA(GetItemD(keys=_pred, index=item_index))] if item_index is not None else []

        if output_nc == 1:
            return Compose(
                select_item_transform + [
                    DTA(ActivationsD(keys=_pred, sigmoid=True)),
                    DTA(AsDiscreteD(keys=_pred, threshold=logit_thresh)) if discrete else lambda x: x,
                    from_engine(_pred),
                ],
                map_items=not decollate,
            )
        else:
            return Compose(
                select_item_transform + [
                    DTA(ActivationsD(keys=_pred, softmax=True)),
                    DTA(AsDiscreteD(keys=_pred, argmax=True, to_onehot=None)) if discrete else lambda x: x,
                    from_engine(_pred),
                ],
                map_items=not decollate,
            )


@ENSEMBLE_TEST_ENGINES.register("segmentation")
class SegmentationEnsembleTestEngine(StrixTestEngine, EnsembleEvaluator):
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
        is_intra_ensemble = isinstance(model_list, (list, tuple)) and len(model_list) > 0
        logger_name = get_attr_(opts, 'logger_name', logger_name)
        self.logger = setup_logger(logger_name)
        crop_size = get_attr_(opts, "crop_size", None)
        use_slidingwindow = opts.slidingwindow
        float_regex = r"=(-?\d+\.\d+).pt"
        decollate = True
        if is_intra_ensemble:
            raise NotImplementedError("Intra ensemble testing not tested yet")
        
        if use_slidingwindow:
            self.logger.info(f"---Use slidingwindow infer!---","\nPatch size: {crop_size}")
        else:
            self.logger.info("---Use simple infer!---")

        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        _image = cfg.get_key("image")
        _pred = cfg.get_key("pred")
        _label = cfg.get_key("label")

        cv_folders = [Path(opts.experiment_path) / f"{i}-th" for i in range(opts.n_fold)]
        cv_folders = filter(lambda x: x.is_dir(), cv_folders)
        best_models = get_models(cv_folders, "best" if use_best_model else "last")
        best_models = list(filter(lambda x: x is not None and x.is_file(), best_models))

        if len(best_models) != opts.n_fold:
            self.logger.warn(
                f"Found {len(best_models)} best models,"
                f"not equal to {opts.n_fold} n_folds.\n"
                f"Use {len(best_models)} best models"
            )
        self.logger.info(f"Using models: {[m.name for m in best_models]}")

        nets = [copy.deepcopy(net),] * len(best_models)

        pred_keys = [f"{_pred}{i}" for i in range(len(best_models))]
        w_ = [float(re.search(float_regex, m.name).group(1)) for m in best_models] if use_best_model else None

        post_transforms = MeanEnsembleD(
            keys=pred_keys,
            output_key=_pred,
            # in this particular example, we use validation metrics as weights
            weights=w_,
        )

        key_val_metric = SegmentationTestEngine.get_metric(opts.phase, opts.output_nc, decollate)
        val_metric_name = list(key_val_metric.keys())[0]
        
        handlers = StrixTestEngine.get_basic_handlers(
            phase=opts.phase,
            out_dir=opts.out_dir,
            model_path=best_models,
            load_dict=[{"net": net} for net in nets],
            logger_name=logger_name,
            stats_dicts={val_metric_name: lambda x: None},
            save_image=opts.save_image,
            image_resample=opts.resample,
            test_loader=test_loader,
            image_batch_transform=from_engine([_image, _image + "_meta_dict"]),
        )
        extra_handlers = SegmentationTestEngine.get_extra_handlers(
            opts=opts, test_loader=test_loader, decollate=decollate, logger_name=logger_name, **kwargs
        )
        if extra_handlers:
            handlers += extra_handlers

        if opts.phase == Phases.TEST_EX:
            prepare_batch_fn = get_unsupervised_prepare_batch_fn(opts, _image, multi_input_keys)
            key_val_metric = None
        elif opts.phase == Phases.TEST_IN:
            prepare_batch_fn = get_prepare_batch_fn(opts, _image, _label, multi_input_keys, multi_output_keys)
        else:
            raise ValueError(f"Got unexpected phase here {opts.phase}, expect testing.")

        if use_slidingwindow:
            inferer = SlidingWindowInferer(roi_size=crop_size, sw_batch_size=opts.n_batch, overlap=0.5)
        else:
            inferer = SimpleInferer()

        EnsembleEvaluator.__init__(
            self,
            device=device,
            val_data_loader=test_loader,
            networks=nets,
            pred_keys=pred_keys,
            prepare_batch=prepare_batch_fn,
            inferer=inferer,
            postprocessing=post_transforms,
            key_val_metric=key_val_metric,
            val_handlers=handlers,
            amp=opts.amp,
            decollate=decollate,
        )
