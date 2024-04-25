import numpy as np
import torch
import torch.nn as nn

#personal modules
from models.blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from models.blocks.attention_blocks import PatchifyVisionMultiheadAttention
from models.blocks.upsampling import InterpolateUpsample, TransposeConv3dUpsample
from models.decoders.conv_decoder import ConvDecoder

class ConvTransDecoder(ConvDecoder):
    def __init__(self,
                 encoder_shapes,
                 num_channels_list,
                 kernel_size=3,
                 activation=nn.ReLU, 
                 normalization=nn.BatchNorm3d,
                 block_type=DoubleConvBlock,
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
        Convolutional decoder for UNet model. We assume that every convolution is a same convolution with no dilation.
        Parameters:
            encoder_shapes (list): list of shapes (N,C,D,H,W) coming from the encoder in the order [skip1, skip2, ..., output]
            num_channels_list (list): list of number of channels in each block
            kernel_size (int or tuple): size of the convolving kernel must be an odd number
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            block_type (blocks.conv_blocks.BaseConvBlock): one the conv blocks inheriting from the BaseConvBlock class
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
        super(ConvTransDecoder, self).__init__(
                                        encoder_shapes,
                                        num_channels_list,
                                        kernel_size=kernel_size,
                                        activation=activation, 
                                        normalization=normalization,
                                        block_type=block_type,
                                        upsampling=upsampling,
                                        skip_mode=skip_mode,
                                        dropout=dropout,
                                    )
        
        #instanciate the encoding conv_layers (conv_blocks + downscaling_layers)
        self.attention_blocks = nn.ModuleList()
        
        self.vision_attention_instanciator = lambda skip_dim, decoder_dim, patch_size: PatchifyVisionMultiheadAttention(
                                                                                    skip_dim, 
                                                                                    decoder_dim, 
                                                                                    patch_size=patch_size, 
                                                                                    embed_size=embed_size, 
                                                                                    num_heads=num_heads,
                                                                                    activation=activation_attention_embedding, 
                                                                                    normalization=normalization_attention,
                                                                                    upscale_attention=upscale_attention,
                                                                                    dropout=dropout_attention,
                                                                                 )
        

        #construction loop initialization
        prev_shape = self.encoder_shapes[0]

        for c_out, skip_shape in zip(self.num_channels_list, self.encoder_shapes[1:]):
        
            #get the patch size
            patch_size= min(list(skip_shape)[2:]) // patch_size_factor

            #attention block
            self.attention_blocks.append(self.vision_attention_instanciator(skip_shape, prev_shape, patch_size))

            #update values
            prev_shape = np.array(skip_shape)
            prev_shape[1] = c_out
            prev_shape = tuple(prev_shape)

            

    def forward(self, x, skips, visualize=False):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size
        skips (list[torch.Tensor]): all the skip connections from the encoder
        visualize (bool): whether or not to return attention weights in visualization format

        Returns:
        x (torch.Tensor): output
        '''
        #attention weights
        attention_weights = []

        #reverse the skips
        skips = skips[::-1]

        #iterate over the number of blocks
        for i in range(self.num_blocks):

            #get skip connection
            skip = skips[i]

            #attention mechanism on the skip connection
            skip, attention_weights_avg = self.attention_blocks[i](skip, x, visualize=visualize)
            attention_weights.append(attention_weights_avg)

            #upscaling
            x = self.upscaling_layers[i](x)

            #add or append mode
            if self.skip_mode == 'append':
                x = torch.cat([skip, x], dim=1)
            elif self.skip_mode == 'add':
                x = x + skip
            else:
                raise NotImplementedError(f"{self.skip_mode} has not been implemented")

            #go through convolutional block
            x = self.conv_blocks[i](x)

        return x, attention_weights