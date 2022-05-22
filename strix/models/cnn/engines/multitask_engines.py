import re
import copy
from pathlib import Path 
import torch
import logging
from strix.configures import config as cfg
from strix.models.cnn.engines import TEST_ENGINES, TRAIN_ENGINES, StrixTestEngine, StrixTrainEngine, ENSEMBLE_TEST_ENGINES
from strix.models.cnn.engines.utils import get_prepare_batch_fn, get_unsupervised_prepare_batch_fn, get_models
from strix.utilities.utils import setup_logger, output_filename_check, get_attr_
from strix.utilities.enum import Phases
from monai_ex.engines import MultiTaskTrainer, SupervisedEvaluatorEx, EnsembleEvaluatorEx
from monai_ex.transforms import MeanEnsembleD, MultitaskMeanEnsembleD
from monai_ex.handlers import EarlyStopHandler, LrScheduleTensorboardHandler, ValidationHandler
from monai_ex.handlers import from_engine_ex as from_engine
from monai_ex.handlers import stopping_fn_from_metric
from monai_ex.inferers import SimpleInfererEx as SimpleInferer


@TRAIN_ENGINES.register("multitask")
class MultiTaskTrainEngine(StrixTrainEngine, MultiTaskTrainer):
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
    ) -> None:
        valid_interval = kwargs.get("valid_interval", 1)
        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        _image = cfg.get_key("image")
        _label = cfg.get_key("label")
        _pred = cfg.get_key("pred")
        _loss = cfg.get_key("loss")
        decollate = False
        freeze_mode = get_attr_(opts, "freeze_mode", None)
        freeze_params = get_attr_(opts, "freeze_params", None)
        logger_name = get_attr_(opts, 'logger_name', logger_name)

        if multi_output_keys is None:
            raise ValueError("No 'multi_output_keys' was specified for MultiTask!")

        subtask1_train_metric = TRAIN_ENGINES[opts.subtask1].get_metric(
            phase=Phases.TRAIN, output_nc=opts.output_nc[0], decollate=decollate, item_index=0, suffix="task1"
        )
        subtask2_train_metric = TRAIN_ENGINES[opts.subtask2].get_metric(
            phase=Phases.TRAIN, output_nc=opts.output_nc[1], decollate=decollate, item_index=1, suffix="task2"
        )

        subtask1_val_metric = TRAIN_ENGINES[opts.subtask1].get_metric(
            phase=Phases.VALID, output_nc=opts.output_nc[0], decollate=decollate, item_index=0, suffix="task1"
        )
        subtask2_val_metric = TRAIN_ENGINES[opts.subtask2].get_metric(
            phase=Phases.VALID, output_nc=opts.output_nc[1], decollate=decollate, item_index=1, suffix="task2"
        )
        val_metric_name = list(subtask1_val_metric.keys())[0]

        task1_tb_image_kwargs = TRAIN_ENGINES[opts.subtask1].get_tensorboard_image_transform(
            opts.output_nc[0], decollate, 0, label_key=multi_output_keys[0]
        )
        task2_tb_image_kwargs = TRAIN_ENGINES[opts.subtask2].get_tensorboard_image_transform(
            opts.output_nc[1], decollate, 1, label_key=multi_output_keys[1]
        )

        val_handlers = StrixTrainEngine.get_basic_handlers(
            phase=Phases.VALID.value,
            model_dir=model_dir,
            net=net,
            optimizer=optim,
            tb_summary_writer=writer,
            logger_name=logger_name,
            stats_dicts={val_metric_name: lambda x: None},
            save_bestmodel=True,
            model_file_prefix=val_metric_name,
            bestmodel_n_saved=opts.save_n_best,
            tensorboard_image_kwargs=[task1_tb_image_kwargs, task2_tb_image_kwargs],
            tensorboard_image_names=["Subtask1", "Subtask2"],
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
            key_val_metric=subtask1_val_metric,
            additional_metrics=subtask2_val_metric,
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
                name=logger_name,
                step_transform=lr_step_transform,
            ),
        ]
        train_handlers += StrixTrainEngine.get_basic_handlers(
            phase=Phases.TRAIN.value,
            model_dir=model_dir,
            net=net,
            optimizer=optim,
            tb_summary_writer=writer,
            logger_name=logger_name,
            stats_dicts={"train_loss": from_engine(_loss, first=True)},
            save_checkpoint=True,
            checkpoint_save_interval=opts.save_epoch_freq,
            ckeckpoint_n_saved=1,
            tensorboard_image_kwargs=[task1_tb_image_kwargs, task2_tb_image_kwargs],
            tensorboard_image_names=["Subtask1", "Subtask2"],
            graph_batch_transform=prepare_batch_fn if opts.visualize else None,
            freeze_mode=freeze_mode,
            freeze_params=freeze_params,
        )

        MultiTaskTrainer.__init__(
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
            key_train_metric=subtask1_train_metric,
            additional_metrics=subtask2_train_metric,
            train_handlers=train_handlers,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict(),
        )
        self.logger = setup_logger(logger_name)


@TEST_ENGINES.register("multitask")
class MultiTaskTestEngine(StrixTestEngine, SupervisedEvaluatorEx):
    def __init__(
        self,
        opts,
        test_loader,
        net,
        device,
        logger_name,
        **kwargs,
    ):
        if opts.slidingwindow:
            raise ValueError("Not implemented yet")

        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        is_supervised = opts.phase == Phases.TEST_IN

        _image = cfg.get_key("image")
        _label = cfg.get_key("label")
        decollate = True
        model_path = opts.model_path[0] if isinstance(opts.model_path, (list, tuple)) else opts.model_path
        logger_name = get_attr_(opts, 'logger_name', logger_name)
        self.logger = setup_logger(logger_name)

        if is_supervised:
            prepare_batch_fn = get_prepare_batch_fn(opts, _image, _label, multi_input_keys, multi_output_keys)
            subtask1_val_metric = TRAIN_ENGINES[opts.subtask1].get_metric(
                phase=opts.phase, output_nc=opts.output_nc[0], decollate=decollate, item_index=0, suffix="task1"
            )
            subtask2_val_metric = TRAIN_ENGINES[opts.subtask2].get_metric(
                phase=opts.phase, output_nc=opts.output_nc[1], decollate=decollate, item_index=1, suffix="task2"
            )
        else:
            prepare_batch_fn = get_unsupervised_prepare_batch_fn(opts, _image, multi_input_keys)
            subtask1_val_metric = subtask2_val_metric = None

        
        handlers = StrixTestEngine.get_basic_handlers(
            phase=opts.phase,
            out_dir=opts.out_dir,
            model_path=model_path,
            load_dict={"net": net},
            logger_name=logger_name,
            stats_dicts={"Metrics": lambda x: None},
            save_image=opts.save_image,
            image_resample=opts.resample,
            test_loader=test_loader,
            image_batch_transform=from_engine([_image, _image + "_meta_dict"]),
        )

        subtask1_extra_handlers = TEST_ENGINES[opts.subtask1].get_extra_handlers(
            opts=opts, test_loader=test_loader, decollate=decollate, logger_name=logger_name, item_index=0, suffix="task1", **kwargs
        )
        subtask2_extra_handlers = TEST_ENGINES[opts.subtask2].get_extra_handlers(
            opts=opts, test_loader=test_loader, decollate=decollate, logger_name=logger_name, item_index=1, suffix="task2", **kwargs
        )

        if subtask1_extra_handlers:
            handlers += subtask1_extra_handlers
        if subtask2_extra_handlers:
            handlers += subtask2_extra_handlers

        SupervisedEvaluatorEx.__init__(
            self,
            device=device,
            val_data_loader=test_loader,
            network=net,
            epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len * len(test_loader)),
            prepare_batch=prepare_batch_fn,
            inferer=SimpleInferer(),
            postprocessing=None,
            key_val_metric=subtask1_val_metric,
            additional_metrics=subtask2_val_metric,
            val_handlers=handlers,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict()
        )


@ENSEMBLE_TEST_ENGINES.register("multitask")
class MultitaskEnsembleTestEngine(StrixTestEngine, EnsembleEvaluatorEx):
    def __init__(self, opts, test_loader, net, device, logger_name, **kwargs):
        if opts.slidingwindow:
            raise ValueError("Not implemented yet")
        
        model_list = opts.model_path
        is_intra_ensemble = isinstance(model_list, (list, tuple)) and len(model_list) > 0
        if is_intra_ensemble:
            raise NotImplementedError()

        use_best_model = kwargs.get("best_val_model", True)
        crop_size = get_attr_(opts, "crop_size", None)
        use_slidingwindow = opts.slidingwindow
        float_regex = r"=(-?\d+\.\d+).pt"
        decollate = True
        is_supervised = opts.phase == Phases.TEST_IN
        logger_name = get_attr_(opts, 'logger_name', logger_name)
        self.logger = setup_logger(logger_name)

        
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

        post_transforms = MultitaskMeanEnsembleD(keys=pred_keys, output_key=_pred, task_num=2, weights=w_)

        if not is_supervised:
            prepare_batch_fn = get_unsupervised_prepare_batch_fn(opts, _image, multi_input_keys)
            subtask1_val_metric = subtask2_val_metric = None
        else:
            prepare_batch_fn = get_prepare_batch_fn(opts, _image, _label, multi_input_keys, multi_output_keys)

            subtask1_val_metric = TRAIN_ENGINES[opts.subtask1].get_metric(
                phase=opts.phase, output_nc=opts.output_nc[0], decollate=decollate, item_index=0, suffix="task1"
            )
            subtask2_val_metric = TRAIN_ENGINES[opts.subtask2].get_metric(
                phase=opts.phase, output_nc=opts.output_nc[1], decollate=decollate, item_index=1, suffix="task2"
            )

        handlers = StrixTestEngine.get_basic_handlers(
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
        subtask1_extra_handlers = TEST_ENGINES[opts.subtask1].get_extra_handlers(
            opts=opts, test_loader=test_loader, decollate=decollate, logger_name=logger_name, item_index=0, suffix="task1", **kwargs
        )
        subtask2_extra_handlers = TEST_ENGINES[opts.subtask2].get_extra_handlers(
            opts=opts, test_loader=test_loader, decollate=decollate, logger_name=logger_name, item_index=1, suffix="task2", **kwargs
        )

        if subtask1_extra_handlers:
            handlers += subtask1_extra_handlers
        if subtask2_extra_handlers:
            handlers += subtask2_extra_handlers
        
        EnsembleEvaluatorEx.__init__(
            self,
            device=device,
            val_data_loader=test_loader,
            networks=nets,
            pred_keys=pred_keys,
            prepare_batch=prepare_batch_fn,
            inferer=SimpleInferer(),
            postprocessing=post_transforms,
            key_val_metric=subtask1_val_metric,
            additional_metrics=subtask2_val_metric,
            val_handlers=handlers,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict()
        )
