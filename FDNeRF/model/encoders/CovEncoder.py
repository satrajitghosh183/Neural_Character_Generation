import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

""" Simple implementation of convolutional encoder for FDNeRF """

class ConvEncoder(nn.Module):
    
    def __init__(self, dim_in = 3, norm_layer = utils.get_norm_layer("group"), padding_type = "reflect", use_leaky_relu = True, use_skip_connection = True, channels = [64, 128, 128]):
        super().__init__()
        self.dim_in = dim_in
        self.norm_layer = norm_layer
        self.padding_type = padding_type
        self.activation = nn.LeakyReLU() if use_leaky_relu else nn.ReLU()
        self.use_skip_connection = use_skip_connection
        
        first_layer_ch, mid_layer_ch, last_layer_ch = channels
        self.n_down_layers = len(channels)
        
        self.conv_in = nn.Sequential(
            nn.Conv2d(dim_in, first_layer_ch, kernel_size = 7, stride = 2, bias = False),
            norm_layer(first_layer_ch),
            self.activation,
        )
        
        channels = first_layer_ch
        for i in range(0, self.n_down_layers):
            conv = nn.Sequential(
                nn.Conv2d(channels, 2*channels, kernel_size = 3, stride = 2, bias = False),
                norm_layer(2*channels),
                self.activation,
            )
            setattr(self, f'conv{i}', conv)
            
            deconv = nn.Sequential(
                nn.ConvTranspose2d(
                    4*channels,
                    channels,
                    kernel_size=3,
                    stride=2,
                    bias = False
                ),
                norm_layer(channels),
                self.activation,
            )
            setattr(self, f'deconv{i}', deconv)
            channels *= 2
            
        self.conv_mid = nn.Sequential(
            nn.Conv2d(channels, mid_layer_ch, kernel_size = 4, stride = 4, bias = False),
            norm_layer(mid_layer_ch),
            self.activation,
        )
        
        self.deconv_last = nn.ConvTranspose2d(
            first_layer_ch,
            last_layer_ch,
            kernel_size = 3,
            stride = 2,
            bias = True
        )
        
        self.dims = [last_layer_ch]
        
    def forward(self,x):
        # Todo: check same_pad_conv2d 
        x = utils.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_in)
        
        x = self.conv_in(x)
        
        inters = []
        for i in range(0, self.n_down_layers):
            conv_i = getattr(self, "conv" + str(i))
            x = utils.same_pad_conv2d(x, padding_type=self.padding_type, layer=conv_i)
            x = conv_i(x)
            inters.append(x)

        x = utils.same_pad_conv2d(x, padding_type=self.padding_type, layer=self.conv_mid)
        x = self.conv_mid(x)
        x = x.reshape(x.shape[0], -1, 1, 1).expand(-1, -1, *inters[-1].shape[-2:])

        for i in reversed(range(0, self.n_down_layers)):
            if self.use_skip_connection:
                x = torch.cat([x, inters[i]], dim=1)
                deconv_i = getattr(self, "deconv" + str(i))
                x = deconv_i(x)
                x = utils.same_pad_conv2d(x,  layer=deconv_i)
        x = self.deconv_last(x)
        x = utils.same_pad_conv2d(x, layer=self.deconv_last)
        return x

if __name__ == "__main__":
    pass