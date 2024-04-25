import torch
import torch.nn as nn
import numpy as np

# personal modules
from models.blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from models.blocks.upsampling import InterpolateUpsample, TransposeConv3dUpsample
from utils.conv_utils import conv3d_output_dim


class ConvHalfDecoder(nn.Module):
    def __init__(self,
                 encoder_shapes,
                 num_channels_list,
                 kernel_size=3,
                 activation=nn.ReLU,
                 normalization=nn.BatchNorm3d,
                 block_type=DoubleConvBlock,
                 upsampling=TransposeConv3dUpsample,
                 dropout=0,
                 channel_ouputconv=64,
                 num_outputconv=2
                 ):
        '''
        Convolutional decoder for UNet model. We assume that every convolution is a same convolution with no dilation.
        Parameters:
            encoder_shapes (list): list of shapes (N,C,D,H,W) coming from the encoder in the order [skip1, skip2, ..., output]
            num_channels_list (list): list of number of channels in each block
            kernel_size (int or tuple): size of the convolving kernel must be an odd number
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            block_type (blocks.conv_blocks.BaseConvBlock): one the conv blocks inheriting from the BaseConvBlock class
            upsampling (blocks.conv.downsampling.Downscale): upsampling scheme
            dropout (float): dropout added to the layer
            channel_ouputconv (int): number of channel at input of convolutionnal layer of the decoder
            num_outputconv (int) : number of convolutional layer at the end of the decoder
        '''
        super(ConvHalfDecoder, self).__init__()

        assert kernel_size % 2 == 1, "kernel size should be an odd number (standard practice)"

        # set number of channels per blocks as well as number of blocks
        self.num_channels_list = num_channels_list
        self.num_blocks = len(num_channels_list)

        # reverse the order of encoder shapes
        self.encoder_shapes = encoder_shapes[::-1]
        assert len(self.encoder_shapes) == self.num_blocks + 1, "the number of blocks plus 1 must be the same as length of the encoder shapes (one encoder shape per block)"

        # conv parameters for same convolution
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # instanciate the encoding upscaling_layers (conv_blocks + downscaling_layers)
        self.upscaling_layers = nn.ModuleList()

        # construction loop initialization
        prev_shape = self.encoder_shapes[0] # output shape
        c_in = prev_shape[1]

        final_shape = self.encoder_shapes[-1]
        for c_out, skip_shape in zip(self.num_channels_list, self.encoder_shapes[1:]):

            # we first do upscaling
            upscale_block = upsampling(prev_shape, skip_shape, c_in, c_out)

            # update values
            c_in = c_out
            prev_shape = skip_shape

            # upsampling added
            self.upscaling_layers.append(upscale_block)

        #resize the channels of the first skip connection
        self.upscaling_layers.append(nn.Conv3d(self.encoder_shapes[-1][1], channel_ouputconv, 1, stride=1))

        self.conv_blocks = nn.ModuleList()
        self.block_instanciator = lambda in_channels, out_channels: block_type(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
        )

        for _ in range(num_outputconv-1):
            self.conv_blocks.append(self.block_instanciator(channel_ouputconv, channel_ouputconv))
            c_in = c_out

        self.conv_blocks.append(self.block_instanciator(channel_ouputconv, self.num_channels_list[-1]))

    def forward(self, x, skips):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size
        skips (list[torch.Tensor]): all the skip connections from the encoder

        Returns:
        x (torch.Tensor): output
        '''


        x = self.upscaling_layers[0](x)

        # reverse the skips
        skips = skips[::-1]

        # iterate over the number of blocks
        for i, skip in enumerate(skips[:]):

            x = x + skip

            x = self.upscaling_layers[i+1](x)

        for layer in self.conv_blocks:
            x = layer(x)

        return x

