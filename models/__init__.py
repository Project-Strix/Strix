import os, time
from utils_cw import check_dir

import torch
from torch.utils.tensorboard import SummaryWriter
from .unet3d import UNet3D
from .unet2d import UNet2D
from .vgg import vgg13_bn, vgg16_bn
from .utils import (print_network, 
                    output_onehot_transform, 
                    LrScheduleTensorboardHandler)

from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.inferers import SimpleInferer, SlidingWindowInferer
from monai.transforms import Compose, Activationsd, AsDiscreted
from ignite.metrics import Accuracy

from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    ValidationHandler,
    LrScheduleHandler,
    CheckpointSaver
)

NETWORK_TYPES = {'CNN':['vgg13','vgg16'], 'FCN':['unet']}

def _get_network_type(name):
    for k, v in NETWORK_TYPES.items():
        if name in v:
            return k

def _get_model_instance(name, tensor_dim):
    return {
        'unet':{'3D': None, '2D': UNet2D},
        'vgg13':{'3D': None, '2D': vgg13_bn},
        'vgg16':{'3D':None, '2D': vgg16_bn}
    }[name][tensor_dim]

def get_network(opts):
                #name, in_channels=1, out_channels=2, is_deconv=False,
                #tensor_dim='2D', layer_order='crb', n_features=64, n_level=4, **kwargs):
    assert hasattr(opts,'model_type') and hasattr(opts,'input_nc') and \
           hasattr(opts, 'tensor_dim') and hasattr(opts,'output_nc')
    
    def get_attr_(obj, name, default):
        return getattr(obj, name) if hasattr(obj, name) else default

    name = opts.model_type
    in_channels, out_channels = opts.input_nc, opts.output_nc
    is_deconv = get_attr_(opts, 'is_deconv', True)
    f_maps = get_attr_(opts, 'n_features', 64)
    n_depth = get_attr_(opts, 'n_level', 4)
    layer_order = get_attr_(opts, 'layer_order', 'crb')
    skip_conn  = get_attr_(opts, 'skip_conn', 'concat')
    n_groups = get_attr_(opts, 'n_groups', 8)
    # bottleneck = get_attr_(opts, 'bottleneck', False)
    # sep_conv   = get_attr_(opts, 'sep_conv', False)

    model = _get_model_instance(name, opts.tensor_dim)

    if name == 'unet':
        model = model(in_channels=in_channels,
                      out_channels=out_channels,
                      f_maps=f_maps,
                      n_level=n_depth,
                      layer_order=layer_order,
                      is_deconv=is_deconv,
                      skip_conn=skip_conn,
                      num_groups=n_groups)
    elif 'vgg' in name:
        model = model(in_channels=in_channels,
                      num_classes=out_channels)
    else:
        raise ValueError(f'Model {name} not available')

    return model

def get_engine(opts, train_loader, test_loader, show_network=True):
    # Print the model type
    print('\nInitialising model {}'.format(opts.model_type))

    framework_type = opts.framework
    device = torch.device("cuda:0")
    model_dir = check_dir(opts.out_dir,'Models')
    # Tensorboard Logger
    writer = SummaryWriter(log_dir=os.path.join(opts.out_dir, 'tensorboard'))
    if not opts.debug:
        os.symlink(os.path.join(opts.out_dir, 'tensorboard'), 
                os.path.join(os.path.dirname(opts.out_dir),'tb', os.path.basename(opts.experiment_name)),target_is_directory=True)

    loss = lr_scheduler = None
    if opts.criterion == 'CE':
        loss = torch.nn.CrossEntropyLoss()
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
    
    if framework_type == 'segmentation':
        raise NotImplementedError
    
    elif framework_type == 'siamese':
        raise NotImplementedError

    elif framework_type == 'selflearning':
        assert _get_network_type(opts.model_type) == 'FCN', f"Only accept FCN arch: {NETWORK_TYPES['FCN']}"
        

    elif framework_type == 'classification':
        assert _get_network_type(opts.model_type) == 'CNN', f"Only accept CNN arch: {NETWORK_TYPES['CNN']}"

        val_handlers = [
            StatsHandler(output_transform=lambda x: None),
            TensorBoardStatsHandler(summary_writer=writer, tag_name="val_acc"),
            CheckpointSaver(save_dir=model_dir, save_dict={"net": net}, save_key_metric=True)
            # TensorBoardImageHandler(summary_writer=writer, batch_transform=lambda x: (x["image"], x["label"]), output_transform=lambda x: x["pred"]),
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