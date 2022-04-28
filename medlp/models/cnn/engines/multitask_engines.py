import torch
from medlp.configures import config as cfg
from medlp.models.cnn.engines import TEST_ENGINES, TRAIN_ENGINES, MedlpTestEngine, MedlpTrainEngine
from medlp.models.cnn.engines.utils import get_prepare_batch_fn, get_unsupervised_prepare_batch_fn
from monai_ex.engines import MultiTaskTrainer, SupervisedEvaluator
from monai_ex.handlers import EarlyStopHandler, LrScheduleTensorboardHandler, ValidationHandler
from monai_ex.handlers import from_engine_ex as from_engine
from monai_ex.handlers import stopping_fn_from_metric
from monai_ex.inferers import SimpleInfererEx as SimpleInferer


@TRAIN_ENGINES.register("multitask")
class MultiTaskTrainEngine(MedlpTrainEngine, MultiTaskTrainer):
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

        if multi_output_keys is None:
            raise ValueError("No 'multi_output_keys' was specified for MultiTask!")

        subtask1_train_metric = TRAIN_ENGINES[opts.subtask1].get_metric(
            phase="train", output_nc=opts.output_nc[0], decollate=decollate, item_index=0, suffix="task1"
        )
        subtask2_train_metric = TRAIN_ENGINES[opts.subtask2].get_metric(
            phase="train", output_nc=opts.output_nc[1], decollate=decollate, item_index=1, suffix="task2"
        )

        subtask1_val_metric = TRAIN_ENGINES[opts.subtask1].get_metric(
            phase="val", output_nc=opts.output_nc[0], decollate=decollate, item_index=0, suffix="task1"
        )
        subtask2_val_metric = TRAIN_ENGINES[opts.subtask2].get_metric(
            phase="val", output_nc=opts.output_nc[1], decollate=decollate, item_index=1, suffix="task2"
        )
        val_metric_name = list(subtask1_val_metric.keys())[0]

        task1_tb_image_kwargs = TRAIN_ENGINES[opts.subtask1].get_tensorboard_image_transform(
            opts.output_nc[0], decollate, 0, label_key=multi_output_keys[0]
        )
        task2_tb_image_kwargs = TRAIN_ENGINES[opts.subtask2].get_tensorboard_image_transform(
            opts.output_nc[1], decollate, 1, label_key=multi_output_keys[1]
        )

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

        evaluator = SupervisedEvaluator(
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
        )

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
        train_handlers += MedlpTrainEngine.get_extra_handlers(
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
            tensorboard_image_kwargs=[task1_tb_image_kwargs, task2_tb_image_kwargs],
            tensorboard_image_names=["Subtask1", "Subtask2"],
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
