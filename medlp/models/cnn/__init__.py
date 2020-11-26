import os, torch
import numpy as np
from functools import partial

from medlp.utilities.handlers import NNIReporterHandler
from medlp.utilities.utils import ENGINES, TEST_ENGINES, assert_network_type, is_avaible_size, output_filename_check
from medlp.models.cnn.utils import output_onehot_transform

from monai.losses import DiceLoss
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.engines import multi_gpu_supervised_trainer
from monai.inferers import SimpleInferer, SlidingWindowClassify, SlidingWindowInferer
from monai.transforms import Compose, Activationsd, AsDiscreted, KeepLargestConnectedComponentd, CastToTyped
from monai.networks import predict_segmentation, one_hot
from monai.utils import Activation, ChannelMatching, Normalisation
from ignite.metrics import Accuracy, MeanSquaredError


from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    MyTensorBoardImageHandler,
    ValidationHandler,
    LrScheduleHandler,
    LrScheduleTensorboardHandler,
    CheckpointSaver,
    CheckpointLoader,
    SegmentationSaver,
    ClassificationSaver,
    MeanDice,
    MetricLogger,
)

@ENGINES.register('segmentation')
def build_segmentation_engine(**kwargs):
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

    assert_network_type(opts.model_type, 'FCN')

    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="val_mean_dice"),
        MyTensorBoardImageHandler(
            summary_writer=writer, 
            batch_transform=lambda x: (x["image"], x["label"]), 
            output_transform=lambda x: x["pred"],
            max_channels=opts.output_nc,
            prefix_name='Val'
        ),
        CheckpointSaver(save_dir=model_dir, save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=3)
    ]
    # If in nni search mode
    if opts.nni: 
        val_handlers += [NNIReporterHandler(metric_name='val_mean_dice', max_epochs=opts.n_epoch, logger_name=logger_name)]
    
    if opts.output_nc == 1:
        trainval_post_transforms = Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
            ]
        )
    else:
        trainval_post_transforms = Compose(
            [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys="pred", to_onehot=True, argmax=True, n_classes=opts.output_nc),
                #KeepLargestConnectedComponentd(keys="pred", applied_labels=[1], independent=False),
            ]
        )

    if opts.criterion in ['CE','WCE']:
        prepare_batch_fn = lambda x : (x["image"], x["label"].squeeze(dim=1))
        if opts.output_nc > 1:
            key_metric_transform_fn = lambda x : (x["pred"], one_hot(x["label"].unsqueeze(dim=1),num_classes=opts.output_nc))
    elif opts.criterion in ['BCE','WBCE']:
        prepare_batch_fn = lambda x : (x["image"], torch.as_tensor(x["label"], dtype=torch.float32))
        if opts.output_nc > 1:
            key_metric_transform_fn = lambda x : (x["pred"], one_hot(x["label"],num_classes=opts.output_nc))
    else:
        prepare_batch_fn = lambda x : (x["image"], x["label"])
        if opts.output_nc > 1:
            key_metric_transform_fn = lambda x : (x["pred"], one_hot(x["label"],num_classes=opts.output_nc))

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len*len(test_loader)),
        prepare_batch=prepare_batch_fn,
        inferer=SimpleInferer(), #SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
        post_transform=trainval_post_transforms,
        key_val_metric={
            "val_mean_dice": MeanDice(include_background=False, output_transform=key_metric_transform_fn)
        },
        val_handlers=val_handlers,
        amp=opts.amp
    )

    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_step_transform = lambda x : evaluator.state.metrics["val_mean_dice"]
    else:
        lr_step_transform = lambda x: ()

    train_handlers = [
        LrScheduleTensorboardHandler(lr_scheduler=lr_scheduler, summary_writer=writer, step_transform=lr_step_transform),
        ValidationHandler(validator=evaluator, interval=valid_interval, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x:x["loss"], name=logger_name),
        CheckpointSaver(save_dir=os.path.join(model_dir,"Checkpoint"), save_dict={"net":net, "optim":optim}, save_interval=opts.save_epoch_freq, epoch_level=True, n_saved=5),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="train_loss", output_transform=lambda x:x["loss"]),
        MyTensorBoardImageHandler(
            summary_writer=writer, batch_transform=lambda x: (x["image"], x["label"]), 
            output_transform=lambda x: x["pred"],
            max_channels=opts.output_nc,
            prefix_name='train'
        ),
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
        post_transform=trainval_post_transforms,
        key_train_metric={
            "train_mean_dice": MeanDice(include_background=False, output_transform=key_metric_transform_fn)
            },
        train_handlers=train_handlers,
        amp=opts.amp
    )
    return trainer


@ENGINES.register('classification')
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

    assert_network_type(opts.model_type, 'CNN')

    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="val_acc"),
        CheckpointSaver(save_dir=os.path.join(model_dir), save_dict={"net": net}, save_key_metric=True, key_metric_n_saved=4),
        MyTensorBoardImageHandler(
            summary_writer=writer, 
            batch_transform=lambda x : (None, None),
            output_transform=lambda x: x["image"],
            prefix_name='Val'
        )
    ]

    train_post_transforms = Compose([
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True)
    ])

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len*len(test_loader)),
        inferer=SimpleInferer(),
        post_transform=train_post_transforms,
        key_val_metric={"val_acc": Accuracy(output_transform=partial(output_onehot_transform,n_classes=opts.output_nc),is_multilabel=True)},
        val_handlers=val_handlers,
        amp=opts.amp
    )

    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_step_transform = lambda x : evaluator.state.metrics["val_acc"]
    else:
        lr_step_transform = lambda x: ()

    train_handlers = [
        LrScheduleTensorboardHandler(lr_scheduler=lr_scheduler, summary_writer=writer, step_transform=lr_step_transform),
        ValidationHandler(validator=evaluator, interval=valid_interval, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"], name=logger_name),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="train_loss", output_transform=lambda x: x["loss"]),
        CheckpointSaver(save_dir=os.path.join(model_dir,"Checkpoint"), save_dict={"net": net, "optim": optim}, save_interval=opts.save_epoch_freq, epoch_level=True, n_saved=5),
        MyTensorBoardImageHandler(
            summary_writer=writer, 
            batch_transform=lambda x : (None, None),
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
        inferer=SimpleInferer(),
        post_transform=train_post_transforms,
        key_train_metric={"train_acc": Accuracy(output_transform=partial(output_onehot_transform,n_classes=opts.output_nc),is_multilabel=True)},
        train_handlers=train_handlers,
        amp=opts.amp
    )

    return trainer


@ENGINES.register('selflearning')
def build_selflearning_engine(**kwargs):
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

    assert_network_type(opts.model_type, 'FCN')

    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="val_loss"),
        MyTensorBoardImageHandler(
            summary_writer=writer, 
            batch_transform=lambda x: (x["image"], x["label"]), 
            output_transform=lambda x: x["pred"],
            max_channels=opts.output_nc,
            prefix_name='Val'
        ),
        CheckpointSaver(save_dir=model_dir, save_dict={"net": net}, save_key_metric=True,
                        key_metric_name='val_mse',key_metric_mode='min',key_metric_n_saved=2)
    ]

    prepare_batch_fn = lambda x : (x["image"], x["label"])
    key_metric_transform_fn = lambda x : (x["pred"], x["label"])

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len*len(test_loader)),
        prepare_batch=prepare_batch_fn,
        inferer=SimpleInferer(),
        key_val_metric={
            "val_mse": MeanSquaredError(key_metric_transform_fn)
        },
        val_handlers=val_handlers,
        amp=opts.amp
    )

    if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        lr_step_transform = lambda x : evaluator.state.metrics["val_mse"]
    else:
        lr_step_transform = lambda x: ()

    train_handlers = [
        LrScheduleTensorboardHandler(lr_scheduler=lr_scheduler, summary_writer=writer, step_transform=lr_step_transform),
        ValidationHandler(validator=evaluator, interval=valid_interval, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x:x["loss"]),
        CheckpointSaver(save_dir=model_dir, save_dict={"net":net, "optim":optim}, save_interval=opts.save_epoch_freq, epoch_level=True, n_saved=5),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="train_loss", output_transform=lambda x:x["loss"]),
        MyTensorBoardImageHandler(
            summary_writer=writer, batch_transform=lambda x: (x["image"], x["label"]), 
            output_transform=lambda x: x["pred"],
            max_channels=opts.output_nc,
            prefix_name='train'
        ),
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
        train_handlers=train_handlers,
        amp=opts.amp
    )
    return trainer


@ENGINES.register('siamese')
def build_siamese_engine(**kwargs):
    opts = kwargs['opts']
    assert_network_type(opts.model_type, 'CNN')

    raise NotImplementedError

@TEST_ENGINES.register('segmentation')
def build_segmentation_test_engine(**kwargs):
    opts = kwargs['opts'] 
    test_loader = kwargs['test_loader'] 
    net = kwargs['net']
    device = kwargs['device'] 
    logger_name = kwargs.get('logger_name', None)
    crop_size = opts.crop_size
    n_batch = opts.n_batch
    use_slidingwindow = is_avaible_size(crop_size)

    if use_slidingwindow:
        print('---Use slidingwindow infer!---')
    else:
        print('---Use simple infer!---')

    assert_network_type(opts.model_type, 'FCN')

    if opts.output_nc == 1:
        post_transforms = Compose(
            [
                Activationsd(keys="pred", sigmoid=True),
                AsDiscreted(keys="pred", threshold_values=True, logit_thresh=0.5),
            ]
        )
    else:
        post_transforms = Compose(
            [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys="pred", argmax=True, n_classes=opts.output_nc),
            ]
        )

    # check output filename
    uplevel = output_filename_check(test_loader.dataset)

    val_handlers = [
        StatsHandler(output_transform=lambda x: None, name=logger_name),
        CheckpointLoader(load_path=opts.model_path, load_dict={"net": net}),
        SegmentationSaver(
            output_dir=opts.experiment_path,
            output_name_uplevel=uplevel,
            batch_transform = lambda x: x['image_meta_dict'],
            #batch_transform=lambda x: {"filename_or_obj": list(map(lambda x: os.path.dirname(x), x["image_meta_dict"]["filename_or_obj"])), "affine":x["image_meta_dict"]["prev_affine"]},
            output_transform=lambda output: output["pred"]
        ),
    ]

    if opts.save_image:
        val_handlers += [
            SegmentationSaver(
                output_dir=opts.experiment_path,
                output_postfix='image',
                output_name_uplevel=uplevel,
                batch_transform = lambda x: x['image_meta_dict'],
                output_transform=lambda output: output["image"]
            )
        ]

    # if opts.criterion == 'CE' or opts.criterion == 'WCE':
    #     prepare_batch_fn = lambda x : (x["image"], x["label"].squeeze(dim=1))
    #     key_metric_transform_fn = lambda x : (x["pred"], x["label"].unsqueeze(dim=1))
    # else:
    #     prepare_batch_fn = lambda x : (x["image"], x["label"])
    #     key_metric_transform_fn = lambda x : (x["pred"], x["label"])

    if opts.phase == 'test_wo_label':
        prepare_batch_fn = lambda x : (x["image"], None)
        key_metric_transform_fn = lambda x : (x["pred"], None)
        key_val_metric = None
    elif opts.phase == 'test':
        prepare_batch_fn = lambda x : (x["image"], x["label"])
        key_metric_transform_fn = lambda x : (x["pred"], x["label"])  
        key_val_metric = {
            "val_mean_dice": MeanDice(include_background=False, output_transform=key_metric_transform_fn)
        }

    inferer = SlidingWindowInferer(roi_size=crop_size, sw_batch_size=n_batch, overlap=0.3) if \
              use_slidingwindow else SimpleInferer()

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        network=net,
        prepare_batch=prepare_batch_fn,
        inferer=inferer,
        post_transform=post_transforms,
        key_val_metric=key_val_metric,
        val_handlers=val_handlers,
        amp=opts.amp
    )

    return evaluator

@TEST_ENGINES.register('classification')
def build_classification_test_engine(**kwargs):
    opts = kwargs['opts'] 
    test_loader = kwargs['test_loader'] 
    net = kwargs['net']
    device = kwargs['device'] 
    logger_name = kwargs.get('logger_name', None)

    assert_network_type(opts.model_type, 'CNN')

    post_transforms = Compose([
        Activationsd(keys="pred", sigmoid=True),
        #AsDiscreted(keys="pred", argmax=True)
    ])

    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        CheckpointLoader(load_path=opts.model_path, load_dict={"net": net}),
        # ClassificationSaver(
        #     output_dir=opts.experiment_path,
        #     output_transform=lambda x : (x['image'], x['pred'].cpu().numpy()),
        #     save_img = True
        # )
        SegmentationSaver(
            output_dir=opts.experiment_path,
            batch_transform=lambda x: {"filename_or_obj":x["image_meta_dict"]["filename_or_obj"] ,"affine":x["affine"]},
            output_transform=lambda output: output["pred"],#[:,0:1,:,:],
        )
    ]

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=test_loader,
        #prepare_batch=lambda x : (x[0]["image"],torch.Tensor(0)),
        network=net,
        inferer=SlidingWindowClassify(roi_size=opts.crop_size, sw_batch_size=4, overlap=0.3),
        post_transform=post_transforms,
        val_handlers=val_handlers,
        amp=opts.amp
    )

    return evaluator