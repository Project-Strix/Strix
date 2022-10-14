from typing import Optional

import torch
from strix.configures import config as cfg
from strix.models.cnn.engines import TRAIN_ENGINES
from strix.models.cnn.engines.engine import StrixTrainEngine
from strix.models.cnn.engines.utils import get_prepare_batch_fn
from strix.utilities.utils import setup_logger, get_attr_
from strix.utilities.transforms import decollate_transform_adaptor as DTA
from strix.utilities.enum import Phases

from monai_ex.engines import SupervisedEvaluatorEx, SupervisedTrainerEx
from monai_ex.inferers import SimpleInferer
from monai_ex.transforms import GetItemD, EnsureTypeD, Compose
from ignite.metrics import MeanSquaredError, SSIM

from monai_ex.handlers import (
    ValidationHandler,
    LrScheduleTensorboardHandler,
    EarlyStopHandler,
    stopping_fn_from_metric,
    from_engine_ex as from_engine,
)


@TRAIN_ENGINES.register("selflearning")
class SelflearningTrainEngine(StrixTrainEngine, SupervisedTrainerEx):
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
        **kwargs
    ):
        valid_interval = kwargs.get("valid_interval", 1)
        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        decollate = False
        logger_name = get_attr_(opts, 'logger_name', logger_name)
        freeze_mode = get_attr_(opts, "freeze_mode", None)
        freeze_params = get_attr_(opts, "freeze_params", None)
        lr_policy_epoch_level = get_attr_(opts, "lr_policy_level", "epoch") == "epoch"
        _image = cfg.get_key("image")
        _label = cfg.get_key("label")
        _loss = cfg.get_key("loss")

        key_val_metric = SelflearningTrainEngine.get_metric(Phases.VALID, opts.output_nc, decollate)
        val_metric_name = list(key_val_metric.keys())[0]
        key_train_metric = SelflearningTrainEngine.get_metric(Phases.TRAIN, opts.output_nc, decollate)

        prepare_batch_fn = get_prepare_batch_fn(opts, _image, _label, multi_input_keys, multi_output_keys)

        val_handlers = StrixTrainEngine.get_basic_handlers(
            phase="val",
            model_dir=model_dir,
            net=net,
            optimizer=optim,
            tb_summary_writer=writer,
            logger_name=logger_name,
            stats_dicts={val_metric_name: lambda x: None},
            save_bestmodel=opts.save_n_best > 0,
            model_file_prefix=val_metric_name,
            bestmodel_n_saved=opts.save_n_best,
            tensorboard_image_kwargs=SelflearningTrainEngine.get_tensorboard_image_transform(
                opts.output_nc, decollate=decollate, interval=opts.tb_dump_img_interval // valid_interval
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
            inferer=SimpleInferer(),
            postprocessing=None,
            key_val_metric=key_val_metric,
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
                epoch_level=lr_policy_epoch_level,
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
            tensorboard_image_kwargs=SelflearningTrainEngine.get_tensorboard_image_transform(
                output_nc=opts.output_nc, decollate=decollate, interval=opts.tb_dump_img_interval
            ),
            graph_batch_transform=prepare_batch_fn if opts.visualize else None,
            freeze_mode=freeze_mode,
            freeze_params=freeze_params,
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
            key_train_metric=key_train_metric,
            train_handlers=train_handlers,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict(),
            ensure_dims=True,
        )
        self.logger = setup_logger(logger_name)

    @staticmethod
    def get_metric(phase: Phases, output_nc: int, decollate: bool, item_index: Optional[int] = None, suffix: str = ''):
        """Return selflearning engine's metrics

        Args:
            phase (Phases): phase enum.
            output_nc (int): output channel num.
            decollate (bool): whether to use decollate.
            item_index (Optional[int], optional): If network's output and label are tuple, specify its index.
             design for multitask compatibility. Defaults to None.
            suffix (str, optional): suffix for subtask. Defaults to ''.
        """
        transform = SelflearningTrainEngine.get_mse_post_transform(decollate, item_index)
        key_metric = MeanSquaredError(transform)
        return {f"{phase.value}_mse_{suffix}": key_metric} if suffix else {f"{phase.value}_mse": key_metric}

    @staticmethod
    def get_mse_post_transform(decollate, item_index):
        _pred = cfg.get_key("pred")
        _label = cfg.get_key("label")

        if item_index:
            transforms = [
                DTA(GetItemD(keys=[_pred, _label], index=item_index)),
                DTA(EnsureTypeD(keys=[_pred, _label], device="cpu")),
                from_engine([_pred, _label])
            ]
        else:
            transforms = [
                DTA(EnsureTypeD(keys=[_pred, _label], device="cpu")),
                from_engine([_pred, _label]),
            ]

        return Compose(transforms, map_items=not decollate)

    @staticmethod
    def get_tensorboard_image_transform(
        output_nc: int, decollate: bool, interval: int, item_index: Optional[int] = None, label_key: Optional[str] = None
    ):
        _image = cfg.get_key("image")
        _label = label_key if label_key else cfg.get_key("label")
        _pred = cfg.get_key("pred")

        post_transform = Compose(
            [
                DTA(GetItemD(keys=_pred, index=item_index)) if item_index is not None else lambda x: x,
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
