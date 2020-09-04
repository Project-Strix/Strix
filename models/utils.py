import logging
from typing import Callable, Optional

import torch
from torch.nn import init
from monai.handlers import LrScheduleHandler
from monai.utils import ensure_tuple, exact_version, optional_import

Events, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Events")
Engine, _ = optional_import("ignite.engine", "0.3.0", exact_version, "Engine")

# def prepare_batch(batchdata, image_key="image", label_key="label"):
#     assert isinstance(batchdata, dict), "default prepare_batch expects dictionary input data."
#     return (
#         (batchdata[image_key], batchdata[label_key])
#         if label_key in batchdata
#         else (batchdata[image_key], None)
#     )

class LrScheduleTensorboardHandler(LrScheduleHandler):
    """
    Ignite handler to update the Learning Rate based on PyTorch LR scheduler.
    """

    def __init__(
        self,
        lr_scheduler,
        summary_writer,
        print_lr: bool = True,
        name: Optional[str] = None,
        epoch_level: bool = True,
        step_transform: Callable = lambda engine: (),
    ):
        super().__init__(
            lr_scheduler = lr_scheduler,
            print_lr = print_lr,
            name = name,
            epoch_level = epoch_level,
            step_transform = step_transform
        )
        self.writer = summary_writer

    def __call__(self, engine):
        args = ensure_tuple(self.step_transform(engine))
        self.lr_scheduler.step(*args)
        if self.print_lr:
            self.logger.info(f"Current learning rate: {self.lr_scheduler._last_lr[0]}")
        if self.writer is not None:
            self.writer.add_scalar("Learning_rate", self.lr_scheduler._last_lr[0], engine.state.iteration)

def output_onehot_transform(output, n_classes=3):
    y_pred, y = output["pred"], output["label"]
    onehot = torch.eye(n_classes)
    #print(onehot[y_pred.squeeze().type(torch.LongTensor)].shape, onehot[y.type(torch.LongTensor)].shape)
    return onehot[y_pred.squeeze().type(torch.LongTensor)], onehot[y.type(torch.LongTensor)]

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
    
def initialize_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)