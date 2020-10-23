import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from .utils import initialize_weights
from .modules import create_feature_maps, Encoder, Decoder_Sum, Decoder_Cat, DoubleConv


class UNet2D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, f_maps=64, n_level=4, 
                 layer_order='crb', is_deconv=True, skip_conn='concat', num_groups=8, 
                 last_act=None, init_weights=True):
        super(UNet2D,self).__init__()
        assert skip_conn in ['sum', 'concat'], "Skip_conn must be one of 'sum,concat'"
        assert last_act in [None, 'sigmoid', 'softmax'], "Last activation must be one of 'None,sigmoid,softmax'"

        if isinstance(f_maps, int):
            # use 4 levels in the encoder path as suggested in the paper
            f_maps = create_feature_maps(f_maps, number_of_fmaps=n_level)

        #! create encoder path consisting of Encoder modules. 
        #! The length of the encoder is equal to `len(f_maps)`
        #! uses DoubleConv as a basic_module for the Encoder
        encoders = []
        pooling = [False] + [True]*len(f_maps)
        f_maps_ = [in_channels] + f_maps
        for i in range(len(f_maps_)-1):
            encoder = Encoder(f_maps_[i], f_maps_[i+1], apply_pooling=pooling[i], basic_module=DoubleConv,
                              conv_layer_order=layer_order, num_groups=num_groups)
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)
        
        #! create decoder path consisting of the Decoder modules. 
        #! The length of the decoder is equal to `len(f_maps) - 1`
        #! uses DoubleConv as a basic_module for the Decoder
        decoders = []
        reversed_f_maps = list(reversed(f_maps))
        for i in range(len(reversed_f_maps)-1):
            if skip_conn == 'sum':
                in_feature_num = reversed_f_maps[i]
                out_feature_num = reversed_f_maps[i + 1]
                decoder = Decoder_Sum(in_feature_num, out_feature_num, basic_module=DoubleConv,
                                      conv_layer_order=layer_order, num_groups=num_groups, is_deconv=is_deconv)
            elif skip_conn == 'concat':
                in_feature_num = reversed_f_maps[i] + reversed_f_maps[i + 1]
                out_feature_num = reversed_f_maps[i + 1]
                decoder = Decoder_Cat(in_feature_num, out_feature_num, basic_module=DoubleConv,
                                      conv_layer_order=layer_order, num_groups=num_groups, is_deconv=is_deconv)
            
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)

        #! in the last layer a 1×1 convolution reduces 
        #! the number of output channels to the number of labels
        self.final_conv = nn.Conv2d(f_maps[0], out_channels, 1)

        self.final_activation = None
        if last_act == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif last_act == 'softmax':
            self.final_activation = nn.Softmax(dim=1)

        if init_weights:
            initialize_weights(self, init_type='kaiming')

    def forward(self,x):
        #! encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        #! remove the last encoder's output from the list. Remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        #! decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output of the previous decoder
            x = decoder(encoder_features, x)

        x = self.final_conv(x)

        if self.final_activation is not None:
            x = self.final_activation(x)

        return x