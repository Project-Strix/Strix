import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers.snip_conv import PrunableConv3d, PrunableDeconv3d

import os, copy, types, time, json
import numpy as np
from utils_cw import Print

def snip_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

def snip_forward_conv3d(self, x):
    #Print('W shape:', self.weight.shape, 'WM shape:', self.weight_mask.shape, color='y')
    return F.conv3d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)

def snip_forward_deconv3d(self, x):
    return F.conv_transpose3d(x, self.weight * self.weight_mask, self.bias,
                              self.stride, self.padding, self.output_padding)

def snip_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)

def apply_prune_mask(net, keep_masks, verbose=False):
    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, PrunableConv3d) or isinstance(layer, PrunableDeconv3d),
        net.modules()
    )
    if verbose:
        [Print(i,'Prune layer:',l, color='y') for i, l in enumerate(prunable_layers)]
    
    for layer, keep_mask in zip(prunable_layers, keep_masks):
        layer.set_pruning_mask(keep_mask)

    return net

def _count_diff(w0, w1):
    assert w0.shape == w1.shape, 'Diff parms shape!'
    diff_num = torch.sum(torch.abs(w0 - w1) > 0)
    return diff_num

def get_pretrain_pruned_unet(opts, in_model, origin_model, channel_mask, verbose=False):
    Print('Copy pretrained weights...', color='g')
    if isinstance(channel_mask[0], torch.Tensor):
        channel_mask = [ c.cpu().numpy().tolist() for c in channel_mask]
    elif isinstance(channel_mask[0], np.ndarray):
        channel_mask = [ c.tolist() for c in channel_mask ]

    new_model = copy.deepcopy(in_model)

    Print('Input chs:', [len(c) for c in channel_mask], color='y')

    layer_idx = 0
    start_mask, end_mask = [1], channel_mask[layer_idx]
    unet_concat_config = {15:1, 12:3, 9:5}
    prev_masks = {}
    for i, (m0, m1) in enumerate(zip(origin_model.modules(), new_model.modules())): #conv->bn order
        keep_prev_mask = True if layer_idx in unet_concat_config.values() else False
        restore_prev_mask = True if layer_idx in unet_concat_config.keys() else False

        if isinstance(m0, nn.BatchNorm3d):
            idx1 = np.squeeze(np.argwhere(end_mask))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))
            Print('  Batch3D channel:', len(end_mask), 'm0 w-shape:', m0.weight.data.shape)
            assert m1.weight.data.shape == m0.weight.data[idx1.tolist()].shape, \
                   'Dim mismatch {}!={}'.format(m1.weight.data.shape, m0.weight.data[idx1.tolist()].shape)
            assert m1.bias.data.shape == m0.bias.data[idx1.tolist()].shape, \
                   'Dim mismatch {}!={}'.format(m1.bias.data.shape, m0.bias.data[idx1.tolist()].shape)

            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.num_batches_tracked = m0.num_batches_tracked.clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_idx += 1
            start_mask = end_mask.copy()
            if layer_idx < len(channel_mask):  # do not change in Final FC
                end_mask = channel_mask[layer_idx]
        elif isinstance(m0, PrunableConv3d) or isinstance(m0, PrunableDeconv3d):
            if keep_prev_mask:
                prev_masks[layer_idx] = [start_mask, end_mask]

            if restore_prev_mask:
                old_mask = prev_masks[unet_concat_config[layer_idx]]
                start_mask = np.concatenate((np.array(old_mask[1]), start_mask))
                Print('Restore prev {}th to {}th: {} -> {}:'.format(unet_concat_config[layer_idx], layer_idx, 
                      len(old_mask[1]), len(start_mask)), color='y', verbose=verbose)
            assert len(start_mask) == m0.weight.data.shape[1] and len(end_mask) == m0.weight.data.shape[0], \
                    'Channel mismatch at {}-{},{}-{}'.format(len(start_mask), m0.weight.data.shape[1], len(end_mask), m0.weight.data.shape[0])

            if isinstance(m0, PrunableConv3d):
                idx0 = np.squeeze(np.argwhere(start_mask))
                idx1 = np.squeeze(np.argwhere(end_mask))
            else:
                idx1 = np.squeeze(np.argwhere(start_mask))
                idx0 = np.squeeze(np.argwhere(end_mask))
            
            Print('OriLayer {}:{}\n  In channel: {:d}->{:d}, Out channel {:d}->{:d}\n'.format(layer_idx, m0, 
                  len(start_mask), idx0.size, len(end_mask), idx1.size), color='g', verbose=verbose)

            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            w1 = m0.weight.data[:, idx0.tolist(), ...].clone()
            w1 = w1[idx1.tolist(), ...].clone()
            
            assert m1.weight.data.shape == w1.shape, \
                   'Weight dim mismatch {}!={}'.format(m1.weight.data.shape, w1.shape)
            m1.weight.data = w1.clone()
            
            if isinstance(m0, PrunableDeconv3d):
                idx1 = idx0

            assert m1.bias.data.shape == m0.bias.data[idx1.tolist()].shape, \
                   'Bias dim mismatch {}!={}'.format(m1.bias.data.shape, m0.bias.data[idx1.tolist()].shape)
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            
            if isinstance(m0, PrunableDeconv3d):
                layer_idx += 1
                start_mask = end_mask.copy()
                if layer_idx < len(channel_mask):
                    end_mask = channel_mask[layer_idx]
        elif isinstance(m0, nn.Conv3d): #last classify conv
            idx0 = np.squeeze(np.argwhere(start_mask))
            w1 = m0.weight.data[:, idx0.tolist(), ...].clone()
            assert m1.weight.data.shape == w1.shape, \
                   'Weight dim mismatch {}!={}'.format(m1.weight.data.shape, w1.shape)
            m1.weight.data = w1.clone()
            m1.bias.data = m0.bias.data.clone() 
    
    return new_model 

def SNIP(input_net, loss_fn, keep_ratio, train_dataloader, use_cuda=True, output_dir=None):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))
    if use_cuda:
        inputs = inputs.cuda().float()
        targets = targets.cuda().byte()

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(input_net)

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, PrunableConv3d) or isinstance(layer, PrunableDeconv3d):
            layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))  
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, PrunableConv3d):
            layer.forward = types.MethodType(snip_forward_conv3d, layer)

        if isinstance(layer, PrunableDeconv3d):
            layer.forward = types.MethodType(snip_forward_deconv3d, layer)

        # if isinstance(layer, nn.Linear):
        #     layer.forward = types.MethodType(snip_forward_linear, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    grads_abs = []
    for layer in net.modules():
        if isinstance(layer, PrunableConv3d) or isinstance(layer, PrunableDeconv3d):
            grads_abs.append(torch.abs(layer.weight_mask.grad))
            #Print('Layer:', layer, 'weight shape:', layer.weight.shape, color='r')
        #if isinstance(layer, nn.BatchNorm3d):
        #    Print('BN:', layer, 'bn shape:', layer.weight.shape, color='g')

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)

    if keep_ratio > 0:
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
    else:
        acceptable_score = np.mean(all_scores)

    keep_masks = []
    for g in grads_abs:
        msk = (g / norm_factor) >= acceptable_score
        if msk.any():
            keep_masks.append(msk.float())
        else:
            onehot = torch.zeros(len(msk))
            keep_masks.append(onehot.scatter_(0, torch.argmax(g), 1).float())

    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))
    Print('Scores min:', torch.min(all_scores), 'Scores max:', torch.max(all_scores), 'Scores mean:', torch.mean(all_scores), color='y')
    if output_dir and os.path.isdir(output_dir):
        np.save(os.path.join(output_dir, 'snip_w_scores_{}.npy'.format(time.strftime("%H%M"))), all_scores.cpu().numpy())

    return keep_masks

def channel_snip_core(self, inputs, targets, net, loss_fn, keep_ratio, min_chs=3, output_dir=None):
    for layer in net.modules():
        if isinstance(layer, PrunableConv3d):
            #Print('Layer w dim:', layer.weight.shape, color='y')
            layer.weight_mask = nn.Parameter(torch.ones([layer.weight.shape[0],1,1,1,1]).cuda()) if use_cuda else \
                                nn.Parameter(torch.ones([layer.weight.shape[0],1,1,1,1]))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
        elif isinstance(layer, PrunableDeconv3d):
            layer.weight_mask = nn.Parameter(torch.ones([1,layer.weight.shape[1],1,1,1]).cuda()) if use_cuda else \
                                nn.Parameter(torch.ones([1,layer.weight.shape[1],1,1,1]))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, PrunableConv3d):
            layer.forward = types.MethodType(snip_forward_conv3d, layer)

        if isinstance(layer, PrunableDeconv3d):
            layer.forward = types.MethodType(snip_forward_deconv3d, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    grads_abs, idx = [], []
    for i, layer in enumerate(net.modules()):
        if isinstance(layer, PrunableConv3d) or isinstance(layer, PrunableDeconv3d):
            grads_abs.append(torch.abs(torch.squeeze(layer.weight_mask.grad)))
            idx.append(i)
            #Print('Layer:', layer, 'weight shape:', layer.weight.shape, color='r')
        #if isinstance(layer, nn.BatchNorm3d):
            #Print('BN:', layer, 'bn shape:', layer.weight.shape, color='g')

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)
    Print('Scores min:', torch.min(all_scores), 'Scores max:', torch.max(all_scores), 'Scores mean:', torch.mean(all_scores), color='y')
    if output_dir and os.path.isdir(output_dir):
        with open(os.path.join(output_dir, 'snip_chs_scores.json'), 'w') as f:
            json.dump(all_scores.cpu().numpy().tolist(), f)

    if keep_ratio > 0:
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
    else:
        acceptable_score = np.mean(all_scores)

    keep_masks = []
    for i, g in enumerate(grads_abs):
        if 1< i < len(grads_abs)-1:
            msk = (g / norm_factor) >= acceptable_score
            if torch.sum(msk) >= min_chs:
                keep_masks.append(msk.cpu().float())
            else:
                ids = torch.topk(g, k=min_chs)[1]
                keep_masks.append(torch.zeros(len(g)).scatter_(0, ids.cpu(), 1))
        else: #keep last conv channel num
            msk = torch.ones(len(g))
            keep_masks.append(msk)

    if output_dir and os.path.isdir(output_dir):
        out_mask = [ m.numpy().tolist() for m in keep_masks]
        with open(os.path.join(output_dir, 'snip_ch_mask_{}.json'.format(keep_ratio)), 'w') as f:
            json.dump(out_mask, f)
    
    remains = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks]))
    Print('Remain #{} channels'.format(remains), color='g')
    return keep_masks

def cSNIP_stepwise(input_net, loss_fn, keep_ratio, train_dataloader, pretrain_weight_file, decay_weight=0.9, min_chs=3, use_cuda=True, output_dir=None):
    net = copy.deepcopy(input_net)

    for i, (inputs, targets) in enumerate(train_dataloader):
        if use_cuda:
            inputs = inputs.cuda()
            targets = targets.cuda()

        mask = channel_snip_core(inputs, targets, net, loss_fn, keep_ratio, min_chs, output_dir)
        
def cSNIP(input_net, loss_fn, keep_ratio, train_dataloader, min_chs=3, use_cuda=True, output_dir=None):
    # TODO: shuffle?

    # Grab a single batch from the training dataset
    inputs, targets = next(iter(train_dataloader))

    # Let's create a fresh copy of the network so that we're not worried about
    # affecting the actual training-phase
    net = copy.deepcopy(input_net)

    if use_cuda:
        inputs = inputs.cuda()
        targets = targets.cuda()
        net = net.cuda()

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    for layer in net.modules():
        if isinstance(layer, PrunableConv3d):
            #Print('Layer w dim:', layer.weight.shape, color='y')
            layer.weight_mask = nn.Parameter(torch.ones([layer.weight.shape[0],1,1,1,1]).cuda()) if use_cuda else \
                                nn.Parameter(torch.ones([layer.weight.shape[0],1,1,1,1]))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
        elif isinstance(layer, PrunableDeconv3d):
            layer.weight_mask = nn.Parameter(torch.ones([1,layer.weight.shape[1],1,1,1]).cuda()) if use_cuda else \
                                nn.Parameter(torch.ones([1,layer.weight.shape[1],1,1,1]))
            nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False

        # Override the forward methods:
        if isinstance(layer, PrunableConv3d):
            layer.forward = types.MethodType(snip_forward_conv3d, layer)

        if isinstance(layer, PrunableDeconv3d):
            layer.forward = types.MethodType(snip_forward_deconv3d, layer)

    # Compute gradients (but don't apply them)
    net.zero_grad()
    outputs = net.forward(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    grads_abs, idx = [], []
    for i, layer in enumerate(net.modules()):
        if isinstance(layer, PrunableConv3d) or isinstance(layer, PrunableDeconv3d):
            grads_abs.append(torch.abs(torch.squeeze(layer.weight_mask.grad)))
            idx.append(i)
            #Print('Layer:', layer, 'weight shape:', layer.weight.shape, color='r')
        #if isinstance(layer, nn.BatchNorm3d):
            #Print('BN:', layer, 'bn shape:', layer.weight.shape, color='g')

    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)
    Print('Scores min:', torch.min(all_scores), 'Scores max:', torch.max(all_scores), 'Scores mean:', torch.mean(all_scores), color='y')
    if output_dir and os.path.isdir(output_dir):
        with open(os.path.join(output_dir, 'snip_chs_scores.json'), 'w') as f:
            json.dump(all_scores.cpu().numpy().tolist(), f)

    if keep_ratio > 0:
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
    else:
        acceptable_score = np.mean(all_scores)

    keep_masks = []
    for i, g in enumerate(grads_abs):
        if 1< i < len(grads_abs)-1:
            msk = (g / norm_factor) >= acceptable_score
            if torch.sum(msk) >= min_chs:
                keep_masks.append(msk.cpu().float())
            else:
                ids = torch.topk(g, k=min_chs)[1]
                keep_masks.append(torch.zeros(len(g)).scatter_(0, ids.cpu(), 1))
        else: #keep last conv channel num
            msk = torch.ones(len(g))
            keep_masks.append(msk)


    if output_dir and os.path.isdir(output_dir):
        out_mask = [ m.numpy().tolist() for m in keep_masks]
        with open(os.path.join(output_dir, 'snip_ch_mask_{}.json'.format(keep_ratio)), 'w') as f:
            json.dump(out_mask, f)
    
    remains = torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks]))
    Print('Remain #{} channels'.format(remains), color='g')
    
    return(keep_masks)
