import os
import re
import logging
import copy
from pathlib import Path
from functools import partial

import torch
from medlp.models.cnn.engines import TRAIN_ENGINES, TEST_ENGINES, ENSEMBLE_TEST_ENGINES
from medlp.utilities.utils import assert_network_type, output_filename_check
from medlp.models.cnn.utils import output_onehot_transform

from monai_ex.engines import SupervisedTrainer, SupervisedEvaluator, EnsembleEvaluator
from monai_ex.inferers import SimpleInferer
from monai_ex.networks import one_hot
from ignite.engine import Events
from ignite.metrics import Accuracy, Precision, Recall
from ignite.handlers import EarlyStopping

from monai_ex.transforms import (
    Compose,
    ActivationsD,
    AsDiscreteD,
    MeanEnsembleD,
    VoteEnsembleD,
    SqueezeDimD
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
    ClassificationSaver,
    ROCAUC,
    stopping_fn_from_metric
)


@TRAIN_ENGINES.register('classification')
def build_classification_engine(**kwargs):
    opts = kwargs['opts']
    train_loader = kwargs['train_loader']
    test_loader = kwargs['test_loader']
    net = kwargs['net']
    loss = kwargs['loss']
    optim = kwargs['optim']
    lr_scheduler = kwargs['lr_scheduler']
    writer = kwargs['writer']
    valid_interval = kwargs['valid_interval']
    device = kwargs['device']
    model_dir = kwargs['model_dir']
    logger_name = kwargs.get('logger_name', None)
    # is_multilabel = opts.output_nc>1

    if opts.criterion in ['BCE', 'WBCE']:
        prepare_batch_fn = lambda x, device, nb : (x["image"].to(device), torch.as_tensor(x["label"].unsqueeze(1), dtype=torch.float32).to(device))
        if opts.output_nc > 1:
            key_metric_transform_fn = lambda x : (x["pred"], one_hot(x["label"],num_classes=opts.output_nc))
    else:
        prepare_batch_fn = lambda x, device, nb : (x["image"].to(device), x["label"].to(device))

    val_metric_name = 'val_auc'
    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="val_acc"),
        CheckpointSaverEx(
            save_dir=model_dir,
            save_dict={"net": net},
            save_key_metric=True,
            key_metric_n_saved=4,
            key_metric_save_after_epoch=100
        ),
        TensorBoardImageHandlerEx(
            summary_writer=writer,
            batch_transform=lambda x: (None, None),
            output_transform=lambda x: x["image"],
            prefix_name='Val'
        )
    ]

    if opts.output_nc == 1:
        train_post_transforms = Compose([
            ActivationsD(keys="pred", sigmoid=True),
            # AsDiscreteD(keys="pred", threshold_values=True, logit_thresh=0.5),
        ])
    else:
        train_post_transforms = Compose([
            ActivationsD(keys="pred", softmax=True),
            AsDiscreteD(keys="pred", argmax=True, to_onehot=False),
            SqueezeDimD(keys='pred')
        ])

    #key_val_metric = Accuracy(output_transform=partial(output_onehot_transform,n_classes=opts.output_nc),is_multilabel=is_multilabel)
    key_val_metric = ROCAUC(output_transform=partial(output_onehot_transform,n_classes=opts.output_nc))

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len*len(test_loader)),
        inferer=SimpleInferer(),
        post_transform=train_post_transforms,
        key_val_metric={val_metric_name: key_val_metric},
        val_handlers=val_handlers,
        amp=opts.amp
    )

    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_step_transform = lambda x: evaluator.state.metrics[val_metric_name]
    else:
        lr_step_transform = lambda x: ()

    train_handlers = [
        LrScheduleTensorboardHandler(lr_scheduler=lr_scheduler, summary_writer=writer, step_transform=lr_step_transform),
        ValidationHandler(validator=evaluator, interval=valid_interval, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"], name=logger_name),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="train_loss", output_transform=lambda x: x["loss"]),
        CheckpointSaverEx(save_dir=os.path.join(model_dir,"Checkpoint"), save_dict={"net": net, "optim": optim}, save_interval=opts.save_epoch_freq, epoch_level=True, n_saved=5), #!n_saved=None
        TensorBoardImageHandlerEx(
            summary_writer=writer,
            batch_transform=lambda x: (None, None),
            output_transform=lambda x: x["image"],
            prefix_name='Train'
        )
    ]

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=opts.n_epoch,
        train_data_loader=train_loader,
        network=net,
        optimizer=optim,
        loss_function=loss,
        epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len*len(train_loader)),
        prepare_batch=prepare_batch_fn,
        inferer=SimpleInferer(),
        post_transform=train_post_transforms,
        key_train_metric={"train_auc": key_val_metric},
        train_handlers=train_handlers,
        amp=opts.amp
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


@TEST_ENGINES.register('classification')
def build_classification_test_engine(**kwargs):
    opts = kwargs['opts']
    test_loader = kwargs['test_loader']
    net = kwargs['net']
    device = kwargs['device']
    logger_name = kwargs.get('logger_name', None)
    is_multilabel = opts.output_nc>1

    assert_network_type(opts.model_name, 'CNN')

    post_transform = Compose([
        ActivationsD(keys="pred", sigmoid=True),
        AsDiscreteD(keys="pred", threshold_values=True, logit_thresh=0.5),
        lambda x : x['pred'].cpu().numpy()
    ])
    if opts.output_nc == 1:
        acc_post_transforms = Compose([
            ActivationsD(keys="pred", sigmoid=True),
            AsDiscreteD(keys="pred", threshold_values=True, logit_thresh=0.5),
            partial(output_onehot_transform,n_classes=opts.output_nc),
        ])
        auc_post_transforms =  Compose([
            ActivationsD(keys="pred", sigmoid=True),
            partial(output_onehot_transform,n_classes=opts.output_nc),
        ])
        
    else:
        acc_post_transforms = Compose([
            ActivationsD(keys="pred", softmax=True),
            AsDiscreteD(keys="pred", argmax=True, to_onehot=False),
            SqueezeDimD(keys="pred"),
            partial(output_onehot_transform,n_classes=opts.output_nc),
        ])
        auc_post_transforms = Compose([
            ActivationsD(keys="pred", softmax=True),
            AsDiscreteD(keys="pred", argmax=True, to_onehot=False),
            SqueezeDimD(keys="pred"),
            partial(output_onehot_transform,n_classes=opts.output_nc),
        ])

    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        CheckpointLoader(load_path=opts.model_path, load_dict={"net": net}),
        ClassificationSaver(
            output_dir=opts.out_dir,
            batch_transform=lambda x : x['image_meta_dict'],
            output_transform=post_transform,
        ),
    ]

    if opts.save_image:
        # check output filename
        uplevel = output_filename_check(test_loader.dataset)
        val_handlers += [
            SegmentationSaver(
                output_dir=opts.out_dir,
                output_postfix='image',
                output_name_uplevel=uplevel,
                resample=False,
                mode="bilinear",
                batch_transform=lambda x: x["image_meta_dict"],
                output_transform=lambda x: x["image"],#[:,0:1,:,:],
            )
        ]


    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        #prepare_batch=lambda x : (x[0]["image"],torch.Tensor(0)),
        network=net,
        inferer=SimpleInferer(), #SlidingWindowClassify(roi_size=opts.crop_size, sw_batch_size=4, overlap=0.3),
        post_transform=None,# post_transforms,
        val_handlers=val_handlers,
        key_val_metric={"test_acc": Accuracy(output_transform=acc_post_transforms,is_multilabel=is_multilabel)},
        additional_metrics={
            'test_auc':ROCAUC(output_transform=auc_post_transforms), 
            'Prec':Precision(output_transform=acc_post_transforms),
            'Recall':Recall(output_transform=acc_post_transforms)
        },
        amp=opts.amp
    )

    return evaluator


@ENSEMBLE_TEST_ENGINES.register('classification')
def build_classification_ensemble_test_engine(**kwargs):
    opts = kwargs['opts']
    test_loader = kwargs['test_loader']
    net = kwargs['net']
    device = kwargs['device']
    best_model = kwargs.get('best_val_model', True)
    logger_name = kwargs.get('logger_name', None)
    logger = logging.getLogger(logger_name)
    is_multilabel = opts.output_nc>1

    cv_folders = [Path(opts.experiment_path)/f'{i}-th' for i in range(opts.n_fold)]
    cv_folders = filter(lambda x: x.is_dir(), cv_folders)
    float_regex = r'=(-?\d+\.\d+).pt'
    int_regex = r'=(\d+).pt'
    if best_model:
        best_models = []
        for folder in cv_folders:
            models = list(filter(lambda x: x.is_file(), [model for model in folder.joinpath('Models').iterdir()]))
            models.sort(key=lambda x: float(re.search(float_regex, x.name).group(1)))
            best_models.append(models[-1])
    else:  # get last
        best_models = []
        for folder in cv_folders:
            models = list(filter(lambda x: x.is_file(), [model for model in (folder/'Models'/'Checkpoint').iterdir()]))
            try:
                models.sort(key=lambda x: int(re.search(int_regex, x.name).group(1)))
            except AttributeError as e:
                invalid_models = list(filter(lambda x: re.search(int_regex, x.name) is None, models))
                print('invalid models:', invalid_models)
                raise e
            best_models.append(models[-1])

    if len(best_models) != opts.n_fold:
        print(
            f'Found {len(best_models)} best models,'
            f'not equal to {opts.n_fold} n_folds.\n'
            f'Use {len(best_models)} best models'
            )
    print(f'Using models: {[m.name for m in best_models]}')

    nets = [copy.deepcopy(net), ]*len(best_models)
    for net, m in zip(nets, best_models):
        CheckpointLoader(load_path=str(m), load_dict={"net": net}, name=logger_name)(None)

    pred_keys = [f"pred{i}" for i in range(len(best_models))]

    # if opts.phase == 'test_wo_label':
    #     output_transform = lambda x: x['pred']
    # else:
    #     output_transform = lambda x: (x['pred'], x['label'])

    if opts.output_nc == 1:  # ensemble_type is 'mean':
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

        acc_post_transforms = Compose([
            ActivationsD(keys="pred", sigmoid=True),
            AsDiscreteD(keys="pred", threshold_values=True, logit_thresh=0.5),
            partial(output_onehot_transform, n_classes=opts.output_nc),
        ])
        auc_post_transforms = Compose([
            ActivationsD(keys="pred", sigmoid=True),
            partial(output_onehot_transform, n_classes=opts.output_nc),
        ])
        ClsSaver_transform = Compose([
            ActivationsD(keys="pred", sigmoid=True),
            AsDiscreteD(keys="pred", threshold_values=True, logit_thresh=0.5),
            lambda x: x['pred'].cpu().numpy()
        ])

    else:  # ensemble_type is 'vote'
        post_transforms = None

        acc_post_transforms = Compose([
            ActivationsD(keys=pred_keys, softmax=True),
            AsDiscreteD(keys=pred_keys, argmax=True, to_onehot=False),
            SqueezeDimD(keys=pred_keys),
            VoteEnsembleD(keys=pred_keys, output_key="pred", num_classes=opts.output_nc),
            partial(output_onehot_transform, n_classes=opts.output_nc),
        ])
        auc_post_transforms = Compose([
            ActivationsD(keys=pred_keys, softmax=True),
            AsDiscreteD(keys=pred_keys, argmax=True, to_onehot=False),
            SqueezeDimD(keys=pred_keys),
            VoteEnsembleD(keys=pred_keys, output_key="pred", num_classes=opts.output_nc),
            partial(output_onehot_transform, n_classes=opts.output_nc),
        ])
        ClsSaver_transform = Compose([
            ActivationsD(keys=pred_keys, softmax=True),
            AsDiscreteD(keys=pred_keys, argmax=True, to_onehot=False),
            SqueezeDimD(keys=pred_keys),
            VoteEnsembleD(keys=pred_keys, output_key="pred", num_classes=opts.output_nc),
            ])


    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        ClassificationSaver(
            output_dir=opts.out_dir,
            batch_transform=lambda x : x['image_meta_dict'],
            output_transform=ClsSaver_transform,
        )
    ]

    if opts.save_image:
        uplevel = output_filename_check(test_loader.dataset)
        val_handlers += [
            SegmentationSaver(
                output_dir=opts.out_dir,
                output_postfix='image',
                output_name_uplevel=uplevel,
                resample=False,
                mode="bilinear",
                batch_transform=lambda x: x["image_meta_dict"],
                output_transform=lambda x: x["image"],
            )
        ]

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

