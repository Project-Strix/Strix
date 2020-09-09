import logging

import numpy as np
import torch
from torch.nn import init

# def prepare_batch(batchdata, image_key="image", label_key="label"):
#     assert isinstance(batchdata, dict), "default prepare_batch expects dictionary input data."
#     return (
#         (batchdata[image_key], batchdata[label_key])
#         if label_key in batchdata
#         else (batchdata[image_key], None)
#     )

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