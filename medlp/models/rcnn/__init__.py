from medlp.utilities.utils import ENGINES, assert_network_type
from medlp.models.rcnn.modeling.detector.generalized_rcnn import GeneralizedRCNN

from monai.engines import RcnnTrainer
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

@ENGINES.register('detection')
def build_detection_engine(**kwargs):
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

    assert_network_type(opts.model_type, 'RCNN')
    
    prepare_batch_fn = lambda x : (x["image"], x["target"])
    key_metric_transform_fn = lambda x : (x["pred"], x["target"])

    train_handlers = [
        LrScheduleTensorboardHandler(lr_scheduler=lr_scheduler, summary_writer=writer),
        #ValidationHandler(validator=evaluator, interval=valid_interval, epoch_level=True),
        StatsHandler(tag_name="train_loss", output_transform=lambda x:x["loss"], name=logger_name),
        CheckpointSaver(save_dir=model_dir, save_dict={"net":net, "optim":optim}, save_interval=opts.save_epoch_freq, epoch_level=True, n_saved=5),
        TensorBoardStatsHandler(summary_writer=writer, tag_name="train_loss", output_transform=lambda x:x["loss"]),
        # MyTensorBoardImageHandler(
        #     summary_writer=writer, 
        #     batch_transform=lambda x: (x["image"], x["label"]), 
        #     output_transform=lambda x: x["pred"],
        #     max_channels=opts.output_nc,
        #     prefix_name='train'
        # ),
    ]

    trainer = RcnnTrainer(
        device=device,
        max_epochs=opts.n_epoch,
        train_data_loader=train_loader,
        network=net,
        optimizer=optim,
        loss_function=loss,
        epoch_length=int(opts.n_epoch_len) if opts.n_epoch_len > 1.0 else int(opts.n_epoch_len*len(train_loader)),
        prepare_batch=prepare_batch_fn,
        post_transform=None,
        train_handlers=train_handlers,
        amp=opts.amp
    )
    return trainer, net