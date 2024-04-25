import torch.nn as nn

from models.segmentation.segmentation import SegmentationModel
from models.encoders.conv_encoder import ConvEncoder
from models.decoders.conv_halfUnet_decoder import ConvHalfDecoder
from models.blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from models.blocks.downsampling import MaxPool3dDownscale, AvgPool3dDownscale
from models.blocks.upsampling import InterpolateUpsample, TransposeConv3dUpsample


class HalfUNet(SegmentationModel):
    def __init__(
            self,
            input_shape,
            num_classes,
            num_channels_list,
            kernel_size=3,
            scale_factor=2,
            activation=nn.ReLU,
            normalization=nn.BatchNorm3d,
            block_type=DoubleConvBlock,
            downsampling=MaxPool3dDownscale,
            upsampling=TransposeConv3dUpsample,
            dropout=0,
            channel_ouputconv=64,
            num_outputconv=2
            ):
        '''
        Implementation of a UNet model
        Parameters:
            input_shape (tuple): (C,D,H,W) of the input
            num_classes (int): number of classes in the segmentation
            num_channels_list (list): list of number of channels in each block
            kernel_size (int or tuple): size of the convolving kernel must be an odd number
            scale_factor (int): factor by which to downscale the image along depth, height and width and then rescale in decoder
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            block_type (blocks.conv_blocks.BaseConvBlock): one the conv blocks inheriting from the BaseConvBlock class
            downsampling (blocks.conv.downsampling.Downscale): downsampling scheme
            upsampling (blocks.conv.downsampling.Downscale): upsampling scheme
            dropout (float): dropout added to the layer
            channel_ouputconv (int): number of channel at input of convolutionnal layer of the decoder
            num_outputconv (int) : number of convolutional layer at the end of the decoder
        '''

        super(HalfUNet, self).__init__()

        # encoder

        #encoder
        self.encoder = ConvEncoder(
                                input_shape,
                                num_channels_list,
                                kernel_size=kernel_size,
                                downscale_factor=scale_factor,
                                activation=activation,
                                normalization=normalization,
                                block_type=block_type,
                                downsampling=downsampling,
                                downscale_last=False,
                                dropout=dropout,
                            )

        # decoder
        self.decoder = ConvHalfDecoder(
            self.encoder.compute_output_dimensions(),
            num_channels_list[-2::-1],
            kernel_size=kernel_size,
            activation=activation,
            normalization=normalization,
            block_type=block_type,
            upsampling=upsampling,
            dropout=dropout,
            channel_ouputconv=channel_ouputconv,
            num_outputconv=num_outputconv
        )

        # ouput layer (channelwise mlp) to have the desired number of classes
        self.output_layer = nn.Conv3d(
            num_channels_list[0],
            num_classes,
            1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        x = self.output_layer(x)
        return x


