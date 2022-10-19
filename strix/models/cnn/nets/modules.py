import torch
from torch import nn
from torch.nn import functional as F
# from inplace_abn import InPlaceABN


class PrunableWeights():
    """Intended to be inherited along with a nn.Module"""

    def set_pruning_mask(self, mask):
        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero

        self.weight.data[mask == 0.] = 0.

        def hook(grads):
            return grads * mask

        self.weight.register_hook(hook)

class PrunableLinear(nn.Linear, PrunableWeights):
    pass

class PrunableConv3d(nn.Conv3d, PrunableWeights):
    pass

class PrunableDeconv3d(nn.ConvTranspose3d, PrunableWeights):
    pass

class PrunableConv2d(nn.Conv2d, PrunableWeights):
    pass

class PrunableDeconv2d(nn.ConvTranspose2d, PrunableWeights):
    pass

def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]

def conv3d(in_channels, out_channels, kernel_size, bias, snip=True, padding=1, stride=1, groups=1, dilation=1):
    conv_func = PrunableConv3d if snip else nn.Conv3d
    return conv_func(in_channels, out_channels, kernel_size, padding=padding, 
                     dilation=dilation, bias=bias, stride=stride, groups=groups)

def conv2d(in_channels, out_channels, kernel_size, bias, snip=True, padding=1, stride=1, groups=1, dilation=1):
    conv_func = PrunableConv2d if snip else nn.Conv2d
    return conv_func(in_channels, out_channels, kernel_size, padding=padding, 
                     dilation=dilation, bias=bias, stride=stride, groups=groups)

def create_conv_2d(in_channels, out_channels, kernel_size, order, 
                   num_groups, padding=1, stride=1, sep_conv=False):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'cB' -> conv + inplace-abn
            'cI' -> conv + inplace-ain
        num_groups (int): number of groups for the GroupNorm
        batch_size (int): batch size for instance_norm init.
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            bias = False #not ('g' in order or 'b' in order)
            if not sep_conv:
                modules.append(('conv', conv2d(in_channels, out_channels, kernel_size, bias, 
                                               padding=padding, stride=stride)))
            else:
                #conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
                #pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
                modules.append(('DepthConv', conv2d(in_channels, in_channels, kernel_size, bias,
                                                    padding=padding, stride=stride, groups=in_channels)))
                modules.append(('PointwiseConv', conv2d(in_channels, out_channels, kernel_size=1, bias=bias,
                                                        stride=1, padding=0, dilation=1, groups=1)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv'
            # number of groups must be less or equal the number of channels
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm2d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm2d(out_channels)))
        elif char == 'B':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'inplace-abn MUST go after the Conv'
            modules.append(('inplace_abn', InPlaceABN(out_channels)))
            #modules.append(('inplace_abn', InPlaceABNSync(out_channels)))
            #modules.append(('inplace_abn', InPlaceABN(out_channels,abn_type='bna')))
        elif char == 'i':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('instancenorm', nn.InstanceNorm2d(in_channels)))
            else:
                modules.append(('instancenorm', nn.InstanceNorm2d(out_channels)))
        else:
            raise ValueError("Unsupported layer type '{}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c', 'B' ]".format(char))

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'ca' -> conv + inplace-abn
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='crb', 
                 num_groups=8, padding=1, **kwargs):
        super(SingleConv, self).__init__()
        stride_downsample = kwargs['stride_ds'] if 'stride_ds' in kwargs.keys() else False
        sep_conv   = kwargs['sep_conv'] if 'sep_conv' in kwargs.keys() else False

        stride = 2 if stride_downsample else 1

        for name, module in create_conv_2d(in_channels, out_channels, kernel_size, 
                                           order, num_groups, padding=padding, 
                                           stride=stride, sep_conv=sep_conv):
            self.add_module(name, module)


class DoubleConv(nn.Sequential):
    """
    A module consisting of two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv2d).
    We use (Conv2d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv2d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        encoder (bool): if True we're in the encoder path, otherwise we're in the decoder
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'ca' -> conv + inplace-abn
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encoder, 
                 kernel_size=3, order='crb', num_groups=8, 
                 stride_ds=False, sep_conv=False):
    
        super(DoubleConv, self).__init__()
        if encoder:
            # we're in the encoder path
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
        else:
            # we're in the decoder path, decrease the number of channels in the 1st convolution
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        # conv1
        self.add_module('SingleConv1',
                        SingleConv(conv1_in_channels, conv1_out_channels, kernel_size, 
                                   order, num_groups, sep_conv=sep_conv))
        # conv2
        self.add_module('SingleConv2',
                        SingleConv(conv2_in_channels, conv2_out_channels, kernel_size, 
                                   order, num_groups, stride_ds=stride_ds, sep_conv=sep_conv))


class Encoder(nn.Module):
    """
    A single module from the encoder path consisting of the optional max
    pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        apply_pooling (bool): if True use MaxPool3d before DoubleConv
        pool_kernel_size (tuple): the size of the window to take a max over
        pool_type (str): pooling layer: 'max' or 'avg'
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, conv_kernel_size=3, apply_pooling=True,
                 pool_kernel_size=(2,2), pool_type='max', basic_module=DoubleConv, 
                 conv_layer_order='crb', num_groups=8, stride_ds=False,sep_conv=False):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool2d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None
       
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=True,
                                         kernel_size=conv_kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         stride_ds=stride_ds,
                                         sep_conv=sep_conv)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.basic_module(x)
        return x


class Decoder_Cat(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=(2,2), 
                 basic_module=DoubleConv, conv_layer_order='crb', num_groups=8, 
                 is_deconv=True, snip=True):
        super(Decoder_Cat, self).__init__()
        deconv_func = PrunableDeconv2d if snip else nn.ConvTranspose2d
        if not is_deconv:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None
        else:
            # otherwise use ConvTranspose3d (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining works correctly
            self.upsample = deconv_func(in_channels-out_channels,
                                        in_channels-out_channels,
                                        kernel_size=kernel_size,
                                        stride=scale_factor,
                                        padding=1,
                                        output_padding=1)

        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            # use nearest neighbor interpolation and summation joining
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
        else:
            # use ConvTranspose and summation joining
            x = self.upsample(x)
        
        x = torch.cat((encoder_features, x), dim=1)
        x = self.basic_module(x)
        return x


class Decoder_Sum(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, scale_factor=(2,2), 
                 basic_module=DoubleConv, conv_layer_order='crb', num_groups=8, is_deconv=True, 
                 snip=True, drop_out=0,sep_conv=False):
        super(Decoder_Sum, self).__init__()
        self.drop_out = drop_out
        deconv_func = PrunableDeconv2d if snip else nn.ConvTranspose2d
        
        if not is_deconv:
            # if DoubleConv is the basic_module use nearest neighbor interpolation for upsampling
            self.upsample = None
        else:
            # otherwise use ConvTranspose (bear in mind your GPU memory)
            # make sure that the output size reverses the MaxPool3d from the corresponding encoder
            # (D_out = (D_in − 1) ×  stride[0] − 2 ×  padding[0] +  kernel_size[0] +  output_padding[0])
            # also scale the number of channels from in_channels to out_channels so that summation joining works correctly
            self.upsample = deconv_func(in_channels,
                                        out_channels,
                                        kernel_size=1,
                                        stride=scale_factor,
                                        padding=0,
                                        output_padding=1) #kenel_sz3:pad1,kenel_sz1:pad0
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = out_channels
        self.dropout_layer = nn.AlphaDropout(self.drop_out)
        self.basic_module = basic_module(in_channels, out_channels,
                                         encoder=False,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         sep_conv=sep_conv)

    def forward(self, encoder_features, x):
        if self.upsample is None:
            # use nearest neighbor interpolation and summation joining
            output_size = encoder_features.size()[2:]
            x = F.interpolate(x, size=output_size, mode='nearest')
        else:
            # use ConvTranspose3d and summation joining
            x = self.upsample(x)
        
        try:
            x.add_(encoder_features)
        except:
            raise ValueError
        
        x = self.basic_module(x)
        if self.drop_out>0:
            x = self.dropout_layer(x)
        return x


