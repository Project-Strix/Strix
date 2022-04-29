from typing import Optional
import re
import logging
import copy
from pathlib import Path


import torch
from medlp.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from medlp.models.cnn.engines.utils import (
    get_models,
    get_prepare_batch_fn,
    get_unsupervised_prepare_batch_fn,
    get_dice_metric_transform_fn,
)
from medlp.utilities.utils import is_avaible_size, output_filename_check, get_attr_
from medlp.utilities.enum import Phases
from medlp.utilities.transforms import decollate_transform_adaptor as DTA
from medlp.configures import config as cfg
from medlp.models.cnn.engines.engine import MedlpTrainEngine, MedlpTestEngine

from monai_ex.inferers import SimpleInfererEx as SimpleInferer, SlidingWindowInferer

from monai_ex.engines import SupervisedTrainerEx, SupervisedEvaluator, EnsembleEvaluator

from monai_ex.transforms import (
    Compose,
    ActivationsD,
    AsDiscreteD,
    MeanEnsembleD,
    GetItemD,
)
from monai_ex.handlers import (
    ValidationHandler,
    LrScheduleTensorboardHandler,
    CheckpointLoader,
    SegmentationSaver,
    MeanDice,
    ROCAUC,
    stopping_fn_from_metric,
    from_engine_ex as from_engine,
    EarlyStopHandler,
)


@TRAIN_ENGINES.register("segmentation")
class SegmentationTrainEngine(MedlpTrainEngine, SupervisedTrainerEx):
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
        _pred = cfg.get_key("pred")
        _loss = cfg.get_key("loss")
        decollate = False

        val_metric = SegmentationTrainEngine.get_metric("val", output_nc=opts.output_nc, decollate=decollate)
        train_metric = SegmentationTrainEngine.get_metric("train", output_nc=opts.output_nc, decollate=decollate)
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
            tensorboard_image_kwargs=SegmentationTrainEngine.get_tensorboard_image_transform(
                output_nc=opts.output_nc, decollate=decollate
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

        evaluator = SupervisedEvaluator(
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
            tensorboard_image_kwargs=SegmentationTrainEngine.get_tensorboard_image_transform(
                output_nc=opts.output_nc, decollate=decollate
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
            inferer=SimpleInferer(),
            postprocessing=None,
            key_train_metric=train_metric,
            train_handlers=train_handlers,
            amp=opts.amp,
            decollate=decollate,
            custom_keys=cfg.get_keys_dict(),
            ensure_dims=True,
        )

    @staticmethod
    def get_metric(phase: str, output_nc: int, decollate: bool, item_index: Optional[int] = None, suffix: str = ''):
        transform = SegmentationTrainEngine.get_dice_post_transform(output_nc, decollate, item_index)
        key_metric = MeanDice(include_background=False, output_transform=transform)
        return {f"{phase}_mean_dice_{suffix}": key_metric} if suffix else {f"{phase}_mean_dice": key_metric}

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
                    DTA(AsDiscreteD(keys=_pred, argmax=True, to_onehot=output_nc)),
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
        output_nc: int, decollate: bool, item_index: Optional[int] = None, label_key: Optional[str] = None
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
                    if item_index is None
                    else from_engine(_pred, transforms=lambda x: x[item_index]),
                ],
                map_items=not decollate,
            )
        else:
            post_transform = Compose(
                [
                    DTA(GetItemD(keys=_pred, index=item_index)) if item_index is not None else lambda x: x,
                    DTA(ActivationsD(keys=_pred, softmax=True)),
                    DTA(AsDiscreteD(keys=_pred, argmax=True, to_onehot=output_nc)),
                    from_engine(_pred)
                    if item_index is None
                    else from_engine(_pred, transforms=lambda x: x[item_index]),
                ],
                map_items=not decollate,
            )

        return {
            "batch_transform": from_engine([_image, _label]),
            "output_transform": post_transform,
            "max_channels": 3,
        }


@TEST_ENGINES.register("segmentation")
class SegmentationTestEngine(MedlpTestEngine, SupervisedEvaluator):
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
        n_batch = opts.n_batch
        resample = opts.resample
        use_slidingwindow = opts.slidingwindow
        multi_input_keys = kwargs.get("multi_input_keys", None)
        multi_output_keys = kwargs.get("multi_output_keys", None)
        _image = cfg.get_key("image")
        _pred = cfg.get_key("pred")
        _label = cfg.get_key("label")
        decollate = True
        model_path = opts.model_path[0] if isinstance(opts.model_path, (list, tuple)) else opts.model_path

        if use_slidingwindow:
            print("---Use slidingwindow infer!---","\nPatch size:", crop_size)
        else:
            print("---Use simple infer!---")

        key_val_metric = SegmentationTestEngine.get_metric(opts.phase, opts.output_nc, decollate)
        val_metric_name = list(key_val_metric.keys())[0]

        val_handlers = [
            SegmentationSaver(
                output_dir=opts.out_dir,
                output_ext=".nii.gz",
                resample=resample,
                data_root_dir=output_filename_check(test_loader.dataset),
                batch_transform=from_engine(_image + "_meta_dict"),
                output_transform=SegmentationTestEngine.get_seg_saver_post_transform(opts.output_nc, decollate),
            ),
        ]

        if opts.save_prob:
            val_handlers += [
                SegmentationSaver(
                    output_dir=opts.out_dir,
                    output_ext=".nii.gz",
                    output_postfix="prob",
                    resample=resample,
                    data_root_dir=output_filename_check(test_loader.dataset),
                    batch_transform=from_engine(_image + "_meta_dict"),
                    output_transform=SegmentationTestEngine.get_seg_saver_post_transform(
                        opts.output_nc, decollate, discrete=False
                    ),
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
            image_batch_transform=from_engine(_image + "_meta_dict"),
            image_output_transform=from_engine(_image),
        )

        if opts.phase == Phases.TEST_EX:
            prepare_batch_fn = get_unsupervised_prepare_batch_fn(opts, _image, multi_input_keys)
            key_val_metric = None
        elif opts.phase == Phases.TEST_IN:
            prepare_batch_fn = get_prepare_batch_fn(opts, _image, _label, multi_input_keys, multi_output_keys)

        if use_slidingwindow:
            inferer = SlidingWindowInferer(roi_size=crop_size, sw_batch_size=n_batch, overlap=0.5)
        else:
            inferer = SimpleInferer()

        SupervisedEvaluator.__init__(
            self,
            device=device,
            val_data_loader=test_loader,
            network=net,
            prepare_batch=prepare_batch_fn,
            inferer=inferer,
            postprocessing=None,
            key_val_metric=key_val_metric,
            val_handlers=val_handlers,
            amp=opts.amp,
            decollate=decollate,
        )

    @staticmethod
    def get_metric(phase: str, output_nc: int, decollate: bool):
        _pred = cfg.get_key("pred")
        _label = cfg.get_key("label")

        transform = SegmentationTrainEngine.get_dice_post_transform(output_nc, decollate)
        if phase == Phases.TEST_EX:
            return {"val_mean_dice": None}
        elif phase == Phases.TEST_IN:
            return {"val_mean_dice": MeanDice(include_background=False, output_transform=transform)}

    @staticmethod
    def get_seg_saver_post_transform(
        output_nc: int,
        decollate: bool,
        discrete: bool = True,
        logit_thresh: float = 0.5,
    ):
        _pred = cfg.get_key("pred")
        _label = cfg.get_key("label")

        if output_nc == 1:
            return Compose(
                [
                    DTA(ActivationsD(keys=_pred, sigmoid=True)),
                    DTA(AsDiscreteD(keys=_pred, threshold=logit_thresh)) if discrete else lambda x: x,
                    from_engine(_pred),
                ],
                map_items=not decollate,
            )
        else:
            return Compose(
                [
                    DTA(ActivationsD(keys=_pred, softmax=True)),
                    DTA(AsDiscreteD(keys=_pred, argmax=True, to_onehot=None)) if discrete else lambda x: x,
                    from_engine(_pred),
                ],
                map_items=not decollate,
            )


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
        CheckpointLoader(load_path=str(m), load_dict={"net": net}, name=logger_name)(None)

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
        key_val_metric={"test_acc": Accuracy(output_transform=acc_post_transforms, is_multilabel=is_multilabel)},
        additional_metrics={
            "test_auc": ROCAUC(output_transform=auc_post_transforms),
            "Prec": Precision(output_transform=acc_post_transforms),
            "Recall": Recall(output_transform=acc_post_transforms),
        },
    )

    return evaluator
