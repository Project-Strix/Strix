import logging

import numpy as np
import torch
import scipy.misc
import scipy.ndimage
import importlib.util
from torch.autograd import Variable
import torch.nn as nn
from torch.optim.lr_scheduler import _LRScheduler
from monai.networks import one_hot
from monai.transforms import (
    Activations, 
    AsDiscrete, 
    SqueezeDim
)

class PolynomialLRDecay(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """
    
    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.:
            raise ValueError('max_decay_steps should be greater than 1.')
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        self.last_step = 0
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_step > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [(base_lr - self.end_learning_rate) * 
                ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                self.end_learning_rate for base_lr in self.base_lrs]
    
    def step(self, step=None):
        if step is None:
            step = self.last_step + 1
        self.last_step = step if step != 0 else 1
        if self.last_step <= self.max_decay_steps:
            decay_lrs = [(base_lr - self.end_learning_rate) * 
                         ((1 - self.last_step / self.max_decay_steps) ** (self.power)) + 
                         self.end_learning_rate for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, decay_lrs):
                param_group['lr'] = lr
        self._last_lr = [group['lr'] for group in self.optimizer.param_groups]

def output_onehot_transform(output, n_classes=3, verbose=False):
    y_pred, y = output["pred"], output["label"]
    if verbose:
        print('Input y_pred:', list(y_pred.cpu().numpy()), '\nInput y_ture:', list(y.cpu().numpy()))
        
    if n_classes == 1:
        return y_pred, y
    
    def onehot_(data, n_class):
        if data.ndimension() == 1:
            data_ = one_hot(data, n_class)
        elif data.ndimension() == 2: #first dim is batch
            data_ = one_hot(data, n_class, dim=1)
        elif data.ndimension() == 3 and data.shape[1] == 1:
            data_ = one_hot(data.squeeze(1), n_class, dim=1)
        else:
            raise ValueError(f'Cannot handle data ndim: {data.ndimension()}, shape: {data.shape}')
        return data_

    pred_ = onehot_(y_pred, n_classes)
    true_ = onehot_(y, n_classes)
    
    assert pred_.shape == true_.shape, f'Pred ({pred_.shape}) and True ({true_.shape}) data have different shape'
    #print('pred, true:', pred_.cpu().numpy(), true_.cpu().numpy())
    return pred_, true_

def acc_output_transform(output, n_classes=2, verbose=False):
    y_pred, y = output["pred"], output["label"]
    pred_ = AsDiscrete(threshold_values=True, logit_thresh=0.5)(y_pred)

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
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


############################################################
#  Pytorch Utility Functions
############################################################

def unique1d(tensor):
    if tensor.size()[0] == 0 or tensor.size()[0] == 1:
        return tensor
    tensor = tensor.sort()[0]
    unique_bool = tensor[1:] != tensor [:-1]
    first_element = Variable(torch.ByteTensor([True]), requires_grad=False)
    if tensor.is_cuda:
        first_element = first_element.cuda()
    unique_bool = torch.cat((first_element, unique_bool),dim=0)
    return tensor[unique_bool.data]


def log2(x):
    """Implementatin of Log2. Pytorch doesn't have a native implemenation."""
    ln2 = Variable(torch.log(torch.FloatTensor([2.0])), requires_grad=False)
    if x.is_cuda:
        ln2 = ln2.cuda()
    return torch.log(x) / ln2


def intersect1d(tensor1, tensor2):
    aux = torch.cat((tensor1, tensor2), dim=0)
    aux = aux.sort(descending=True)[0]
    return aux[:-1][(aux[1:] == aux[:-1]).data]


def shem(roi_probs_neg, negative_count, ohem_poolsize):
    """stochastic hard example mining: from a list of indices (referring to non-matched predictions),
        determine a pool of highest scoring (worst false positives) of size negative_count*ohem_poolsize.
        Then, sample n (= negative_count) predictions of this pool as negative examples for loss.

    :param roi_probs_neg: tensor of shape (n_predictions, n_classes).
    :param negative_count: int.
    :param ohem_poolsize: int.
    :return: (negative_count).  indices refer to the positions in roi_probs_neg. If pool smaller than expected due to
    limited negative proposals availabel, this function will return sampled indices of number < negative_count without
    throwing an error.
    """
    # sort according to higehst foreground score.
    probs, order = roi_probs_neg[:, 1:].max(1)[0].sort(descending=True)
    select = torch.tensor((ohem_poolsize * int(negative_count), order.size()[0])).min().int()
    pool_indices = order[:select]
    rand_idx = torch.randperm(pool_indices.size()[0])
    return pool_indices[rand_idx[:negative_count].cuda()]


def initialize_weights(net):
    """
   Initialize model weights. Current Default in Pytorch (version 0.4.1) is initialization from a uniform distriubtion.
   Will expectably be changed to kaiming_uniform in future versions.
   """
    init_type = net.cf.weight_init

    for m in [module for module in net.modules() if type(module) in [nn.Conv2d, nn.Conv3d,
                                                                     nn.ConvTranspose2d,
                                                                     nn.ConvTranspose3d,
                                                                     nn.Linear]]:
        if init_type == 'xavier_uniform':
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        elif init_type == 'xavier_normal':
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()

        elif init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(m.weight.data, mode='fan_out', nonlinearity=net.cf.relu, a=0)
            if m.bias is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / np.sqrt(fan_out)
                nn.init.uniform_(m.bias, -bound, bound)

        elif init_type == "kaiming_normal":
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity=net.cf.relu, a=0)
            if m.bias is not None:
                fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                bound = 1 / np.sqrt(fan_out)
                nn.init.normal_(m.bias, -bound, bound)


class NDConvGenerator(object):
    """
    generic wrapper around conv-layers to avoid 2D vs. 3D distinguishing in code.
    """
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, c_in, c_out, ks, pad=0, stride=1, norm=None, relu='relu'):
        """
        :param c_in: number of in_channels.
        :param c_out: number of out_channels.
        :param ks: kernel size.
        :param pad: pad size.
        :param stride: kernel stride.
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :return: convolved feature_map.
        """
        if self.dim == 2:
            conv = nn.Conv2d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm2d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm2d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented...')
                conv = nn.Sequential(conv, norm_layer)

        else:
            conv = nn.Conv3d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
            if norm is not None:
                if norm == 'instance_norm':
                    norm_layer = nn.InstanceNorm3d(c_out)
                elif norm == 'batch_norm':
                    norm_layer = nn.BatchNorm3d(c_out)
                else:
                    raise ValueError('norm type as specified in configs is not implemented... {}'.format(norm))
                conv = nn.Sequential(conv, norm_layer)

        if relu is not None:
            if relu == 'relu':
                relu_layer = nn.ReLU(inplace=True)
            elif relu == 'leaky_relu':
                relu_layer = nn.LeakyReLU(inplace=True)
            else:
                raise ValueError('relu type as specified in configs is not implemented...')
            conv = nn.Sequential(conv, relu_layer)

        return conv


def get_one_hot_encoding(y, n_classes):
    """transform a numpy label array to a one-hot array of the same shape.

    :param y: array of shape (b, 1, y, x, (z)).
    :param n_classes: int, number of classes to unfold in one-hot encoding.
    :return y_ohe: array of shape (b, n_classes, y, x, (z))
    """
    dim = len(y.shape) - 2
    if dim == 2:
        y_ohe = np.zeros((y.shape[0], n_classes, y.shape[2], y.shape[3])).astype('int32')
    if dim ==3:
        y_ohe = np.zeros((y.shape[0], n_classes, y.shape[2], y.shape[3], y.shape[4])).astype('int32')
    for cl in range(n_classes):
        y_ohe[:, cl][y[:, 0] == cl] = 1
    return y_ohe


def get_dice_per_batch_and_class(pred, y, n_classes):
    '''computes dice scores per batch instance and class.

    :param pred: prediction array of shape (b, 1, y, x, (z)) (e.g. softmax prediction with argmax over dim 1)
    :param y: ground truth array of shape (b, 1, y, x, (z)) (contains int [0, ..., n_classes]
    :param n_classes: int
    :return: dice scores of shape (b, c)
    '''
    pred = get_one_hot_encoding(pred, n_classes)
    y = get_one_hot_encoding(y, n_classes)
    axes = tuple(range(2, len(pred.shape)))
    intersect = np.sum(pred*y, axis=axes)
    denominator = np.sum(pred, axis=axes)+np.sum(y, axis=axes) + 1e-8
    dice = 2.0*intersect / denominator
    return dice


def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes)
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(int(ax))
    return input


def batch_dice(pred, y, false_positive_weight=1.0, smooth=1e-6):
    '''compute soft dice over batch. this is a differentiable score and can be used as a loss function.
        only dice scores of foreground classes are returned, since training typically
        does not benefit from explicit background optimization. Pixels of the entire batch are considered a pseudo-volume to compute dice scores of.
        This way, single patches with missing foreground classes can not produce faulty gradients.

    :param pred: (b, c, y, x, (z)), softmax probabilities (network output). (c==classes)
    :param y: (b, c, y, x, (z)), one-hot-encoded segmentation mask.
    :param false_positive_weight: float [0,1]. For weighting of imbalanced classes,
    reduces the penalty for false-positive pixels. Can be beneficial sometimes in data with heavy fg/bg imbalances.
    :return: soft dice score (float). This function discards the background score and returns the mean of foreground scores.
    '''
    if len(pred.size()) == 4:
        axes = (0, 2, 3)
        intersect = sum_tensor(pred * y, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred + y, axes, keepdim=False)
        return torch.mean(( (2 * intersect + smooth) / (denom + smooth) )[1:]) # only fg dice here.

    elif len(pred.size()) == 5:
        axes = (0, 2, 3, 4)
        intersect = sum_tensor(pred * y, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred + y, axes, keepdim=False)
        return torch.mean(( (2*intersect + smooth) / (denom + smooth) )[1:]) # only fg dice here.

    else:
        raise ValueError('wrong input dimension in dice loss')



def batch_dice_mask(pred, y, mask, false_positive_weight=1.0, smooth=1e-6):
    ''' compute soft dice over batch. this is a diffrentiable score and can be used as a loss function.
        only dice scores of foreground classes are returned, since training typically
        does not benefit from explicit background optimization. Pixels of the entire batch are considered a pseudo-volume to compute dice scores of.
        This way, single patches with missing foreground classes can not produce faulty gradients.

    :param pred: (b, c, y, x, (z)), softmax probabilities (network output).
    :param y: (b, c, y, x, (z)), one hote encoded segmentation mask.
    :param false_positive_weight: float [0,1]. For weighting of imbalanced classes,
    reduces the penalty for false-positive pixels. Can be beneficial sometimes in data with heavy fg/bg imbalances.
    :return: soft dice score (float). This function discards the background score and returns the mean of foreground scores.
    '''

    mask = mask.unsqueeze(1).repeat(1, 2, 1, 1)

    if len(pred.size()) == 4:
        axes = (0, 2, 3)
        intersect = sum_tensor(pred * y * mask, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred * mask + y * mask, axes, keepdim=False)
        return torch.mean(( (2*intersect + smooth) / (denom + smooth))[1:]) # only fg dice here.

    elif len(pred.size()) == 5:
        axes = (0, 2, 3, 4)
        intersect = sum_tensor(pred * y, axes, keepdim=False)
        denom = sum_tensor(false_positive_weight*pred + y, axes, keepdim=False)
        return torch.mean(( (2*intersect + smooth) / (denom + smooth) )[1:]) # only fg dice here.

    else:
        raise ValueError('wrong input dimension in dice loss')

