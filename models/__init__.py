import os, time
from types import SimpleNamespace as sn
from utils_cw import check_dir

import torch
from torch.utils.tensorboard import SummaryWriter
from .unet3d import UNet3D
from .unet2d import UNet2D
from .vgg import vgg13_bn, vgg16_bn
from .resnet import resnet34
from .utils import print_network, output_onehot_transform

from monai.losses import DiceLoss
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.inferers import SimpleInferer, SlidingWindowClassify
from monai.transforms import Compose, Activationsd, AsDiscreted, KeepLargestConnectedComponentd
from monai.networks.nets import UNet
from ignite.metrics import Accuracy

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
    MeanDice
)

NETWORK_TYPES = {'CNN':['vgg13','vgg16','resnet34'], 'FCN':['unet']}

def _get_network_type(name):
    for k, v in NETWORK_TYPES.items():
        if name in v:
            return k

def _get_model_instance(name, tensor_dim):
    return {
        'unet':{'3D': None, '2D': UNet2D},
        'vgg13':{'3D': None, '2D': vgg13_bn},
        'vgg16':{'3D': None, '2D': vgg16_bn},
        'resnet34':{'3D': None, '2D': resnet34}
    }[name][tensor_dim]

def get_network(opts):
    assert hasattr(opts,'model_type') and hasattr(opts,'input_nc') and \
           hasattr(opts, 'tensor_dim') and hasattr(opts,'output_nc')
    
    def get_attr_(obj, name, default):
        return getattr(obj, name) if hasattr(obj, name) else default

    name = opts.model_type
    in_channels, out_channels = opts.input_nc, opts.output_nc
    is_deconv = get_attr_(opts, 'is_deconv', False)
    f_maps = get_attr_(opts, 'n_features', 64)
    n_depth = get_attr_(opts, 'n_level', 4)
    layer_order = get_attr_(opts, 'layer_order', 'crb')
    skip_conn  = get_attr_(opts, 'skip_conn', 'concat')
    n_groups = get_attr_(opts, 'n_groups', 8)
    # bottleneck = get_attr_(opts, 'bottleneck', False)
    # sep_conv   = get_attr_(opts, 'sep_conv', False)

    model = _get_model_instance(name, opts.tensor_dim)

    dim = 2 if opts.tensor_dim == '2D' else 3
    if name == 'unet':
        model = UNet(
            dimensions=dim,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=(32, 64, 128, 265, 512),
            strides=(2, 2, 2, 2),
            num_res_units=0,
            act='relu',
            norm='batch'
        )
        # model = model(in_channels=in_channels,
        #               out_channels=out_channels,
        #               f_maps=f_maps,
        #               n_level=n_depth,
        #               layer_order=layer_order,
        #               is_deconv=is_deconv,
        #               skip_conn=skip_conn,
        #               num_groups=n_groups)
    elif 'vgg' in name:
        model = model(in_channels=in_channels,
                      num_classes=out_channels)
    elif 'resnet' in name:
        model = model(in_channels=in_channels,
                      num_classes=out_channels)
    else:
        raise ValueError(f'Model {name} not available')  

    return model


def get_engine(opts, train_loader, test_loader, show_network=True):
    # Print the model type
    print('\nInitialising model {}'.format(opts.model_type))

    framework_type = opts.framework
    device = torch.device("cuda:0") if opts.gpus != '-1' else torch.device("cpu")
    model_dir = check_dir(opts.out_dir,'Models')
    # Tensorboard Logger
    writer = SummaryWriter(log_dir=os.path.join(opts.out_dir, 'tensorboard'))
    if not opts.debug:
        os.symlink(os.path.join(opts.out_dir, 'tensorboard'), 
                os.path.join(os.path.dirname(opts.out_dir),'tb', os.path.basename(opts.experiment_name)),target_is_directory=True)

    loss = lr_scheduler = None
    if opts.criterion == 'CE':
        loss = torch.nn.CrossEntropyLoss()
    elif opts.criterion == 'MSE':
        loss = torch.nn.MSELoss()
    elif opts.criterion == 'DCE':
        loss = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
    else:
        raise NotImplementedError

    net = get_network(opts).to(device)
    if show_network:
        print_network(net)

    if opts.optim == 'adam':
        optim = torch.optim.Adam(net.parameters(), opts.lr)
    elif opts.optim == 'sgd':
        optim = torch.optim.SGD(net.parameters(), opts.lr)
    else:
        raise NotImplementedError
    
    if opts.lr_policy == 'const':
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda x:1)
    elif opts.lr_policy == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 
                                                        step_size=opts.lr_policy_params['step_size'], 
                                                        gamma=opts.lr_policy_params['gamma'])
    elif opts.lr_policy == 'SGDR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 
                                                                            T_0=opts.lr_policy_params['t0'],
                                                                            T_mult=opts.lr_policy_params['T_mult'],
                                                                            eta_min=opts.lr_policy_params['eta_min'])
    #! Segmentation module
    if framework_type == 'segmentation':
        assert _get_network_type(opts.model_type) == 'FCN', f"Only accept FCN arch: {NETWORK_TYPES['FCN']}"

        val_handlers = [
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(summary_writer=writer, tag_name="val_mean_dice"),
            MyTensorBoardImageHandler(
                summary_writer=writer, 
                batch_transform=lambda x: (x["image"], x["label"]), 
                output_transform=lambda x: x["pred"],
                max_channels=opts.output_nc,
                prefix_name='Val'
            ),
            CheckpointSaver(save_dir=model_dir, save_dict={"net": net}, save_key_metric=True)
        ]

        trainval_post_transforms = Compose(
            [
                Activationsd(keys="pred", softmax=True),
                AsDiscreted(keys="pred", to_onehot=True, argmax=True, n_classes=opts.output_nc),
                #KeepLargestConnectedComponentd(keys="pred", applied_labels=[1], independent=False),
            ]
        )

        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=test_loader,
            network=net,
            inferer=SimpleInferer(), #SlidingWindowInferer(roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5),
            post_transform=trainval_post_transforms,
            key_val_metric={
                "val_mean_dice": MeanDice(include_background=True, to_onehot_y=True, output_transform=lambda x: (x["pred"], x["label"]))
            },
            val_handlers=val_handlers,
        )

        train_handlers = [
            LrScheduleTensorboardHandler(lr_scheduler=lr_scheduler, summary_writer=writer),
            ValidationHandler(validator=evaluator, interval=5, epoch_level=True),
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
            inferer=SimpleInferer(),
            amp=False,
            post_transform=trainval_post_transforms,
            key_train_metric={"train_mean_dice": MeanDice(include_background=False, to_onehot_y=True, output_transform=lambda x: (x["pred"], x["label"]))},
            train_handlers=train_handlers,
        )
        return trainer
    
    #! Siamese moduel
    elif framework_type == 'siamese':
        raise NotImplementedError
    
    #! Selflearning module
    elif framework_type == 'selflearning':
        assert _get_network_type(opts.model_type) == 'FCN', f"Only accept FCN arch: {NETWORK_TYPES['FCN']}"

        val_handlers = [
            StatsHandler(output_transform=lambda x: None),
            CheckpointSaver(save_dir=model_dir, save_dict={"net": net}, save_key_metric=True)
        ]

        raise NotImplementedError
       
    #! Classification module
    elif framework_type == 'classification':
        assert _get_network_type(opts.model_type) == 'CNN', f"Only accept CNN arch: {NETWORK_TYPES['CNN']}"

        val_handlers = [
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(summary_writer=writer, tag_name="val_acc"),
            CheckpointSaver(save_dir=model_dir, save_dict={"net": net}, save_key_metric=True)
        ]

        train_post_transforms = Compose([
            Activationsd(keys="pred", softmax=True),
            AsDiscreted(keys="pred", argmax=True)
        ])

        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=test_loader,
            network=net,
            inferer=SimpleInferer(),
            post_transform=train_post_transforms,
            key_val_metric={"val_acc": Accuracy(output_transform=output_onehot_transform,is_multilabel=True)},
            val_handlers=val_handlers,
        )

        train_handlers = [
            LrScheduleTensorboardHandler(lr_scheduler=lr_scheduler, summary_writer=writer),
            ValidationHandler(validator=evaluator, interval=5, epoch_level=True),
            StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
            TensorBoardStatsHandler(summary_writer=writer, tag_name="train_loss", output_transform=lambda x: x["loss"]),
            CheckpointSaver(save_dir=model_dir, save_dict={"net": net, "optim": optim}, save_interval=opts.save_epoch_freq, epoch_level=True, n_saved=5),
        ]

        trainer = SupervisedTrainer(
            device=device,
            max_epochs=opts.n_epoch,
            train_data_loader=train_loader,
            network=net,
            optimizer=optim,
            loss_function=loss,
            inferer=SimpleInferer(),
            amp=False,
            post_transform=train_post_transforms,
            key_train_metric={"train_acc": Accuracy(output_transform=output_onehot_transform,is_multilabel=True)},
            train_handlers=train_handlers,
        )

        return trainer


def get_test_engine(opts, test_loader):
    framework_type = opts.framework
    device = torch.device("cuda:0") if opts.gpus != '-1' else torch.device("cpu")

    net = get_network(opts).to(device)

    if framework_type == 'classification':
        post_transforms = Compose([
            Activationsd(keys="pred", softmax=True),
            #AsDiscreted(keys="pred", argmax=True)
        ])

        val_handlers = [
            StatsHandler(output_transform=lambda x: None),
            CheckpointLoader(load_path=opts.model_path, load_dict={"net": net}),
            SegmentationSaver(
                output_dir=opts.out_dir,
                batch_transform=lambda x: {"filename_or_obj":x["image_meta_dict"]["filename_or_obj"] ,"affine":x["affine"]},
                output_transform=lambda output: output["pred"][:,0:1,:,:],
            )
        ]

        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=test_loader,
            network=net,
            inferer=SlidingWindowClassify(roi_size=opts.crop_size, sw_batch_size=4, overlap=0.5),
            post_transform=post_transforms,
            val_handlers=val_handlers
        )

        return evaluator
    elif framework_type == 'segmentation':
        pass