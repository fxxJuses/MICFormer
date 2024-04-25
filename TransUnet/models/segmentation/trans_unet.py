import torch.nn as nn 

from models.segmentation.unet import UNet
from models.encoders.conv_encoder import ConvEncoder
from models.decoders.conv_trans_decoder import ConvTransDecoder
from models.blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from models.blocks.downsampling import MaxPool3dDownscale, AvgPool3dDownscale
from models.blocks.upsampling import InterpolateUpsample, TransposeConv3dUpsample

class TransUNet(UNet):
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
            patch_size_factor=8,
            embed_size=64, 
            num_heads=8,
            activation_attention_embedding=nn.Identity,
            normalization_attention=nn.Identity,
            upscale_attention=TransposeConv3dUpsample,
            skip_mode='append',
            dropout=0,
            dropout_attention=0,
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
            patch_size_factor (int): amount by which the smallest dimension is divided
            embed_size (int): size of the attention layer
            num_heads (int): number of attention heads (dimension of each head is embed_size // num_heads)
            activation_attention_embedding (def None -> torch.nn.Module): activation used by the embedding layer in the attention block
            normalization_attention (def int -> torch.nn.modules.batchnorm._NormBase): normalization in embedding layer in the attention block
            upscale_attention (blocks.conv.downsampling.Downscale): upsampling scheme for attention output
            skip_mode (str): one of 'append' | 'add' refers to how the skip connection is added back to the decoder path
            dropout (float): dropout added to the layer
            dropout_attention (float): dropout added to the embedding layer in the attention block
        '''
        super(TransUNet, self).__init__(
            input_shape, 
            num_classes,
            num_channels_list,
            kernel_size=kernel_size,
            scale_factor=scale_factor,
            activation=activation, 
            normalization=normalization,
            block_type=block_type,
            downsampling=downsampling,
            upsampling=upsampling,
            skip_mode=skip_mode,
            dropout=dropout,
        )
        
        #decoder
        self.decoder = ConvTransDecoder(
                            self.encoder.compute_output_dimensions(),
                            num_channels_list[-2::-1],
                            kernel_size=kernel_size,
                            activation=activation, 
                            normalization=normalization,
                            block_type=block_type,
                            upsampling=upsampling,
                            patch_size_factor=patch_size_factor,
                            embed_size=embed_size, 
                            num_heads=num_heads,
                            activation_attention_embedding=activation_attention_embedding,
                            normalization_attention=normalization_attention,
                            upscale_attention=upscale_attention,
                            skip_mode=skip_mode,
                            dropout=dropout,
                            dropout_attention=dropout_attention,
                        )
        
    
    def forward(self, x, visualize=False):
        x, skip_connections = self.encoder(x)
        x, attention_weights = self.decoder(x, skip_connections, visualize=visualize)
        x = self.output_layer(x)

        #if visualization is set to true then return attention_weights
        if not visualize:
            return x
        else:
            return x, attention_weights