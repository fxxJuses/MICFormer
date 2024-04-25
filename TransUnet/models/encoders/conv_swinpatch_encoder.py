import torch.nn as nn

# personal modules
import torch
from models.blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from models.blocks.downsampling import MaxPool3dDownscale, AvgPool3dDownscale
from utils.conv_utils import conv3d_output_dim
from models.encoders.conv_encoder import ConvEncoder

class ConvPatchEncoder(nn.Module):
    def __init__(self,
                 input_shape,
                 num_channels_list,
                 channel_embedding,
                 kernel_size=3,
                 downscale_factor=2,
                 activation=nn.ReLU,
                 normalization=nn.BatchNorm3d,
                 block_type=DoubleConvBlock,
                 downsampling=MaxPool3dDownscale,
                 downscale_last=False,
                 dropout=0,
                 patch_size=3,
                 ):
        '''
        Convolutional encoder for UNet model. We assume that every convolution is a same convolution with no dilation.
        Parameters:
            input_shape (tuple): (C,D,H,W) of the input
            num_channels_list (list): list of number of channels in each block
            kernel_size (int or tuple): size of the convolving kernel must be an odd number
            downscale_factor (int): factor by which to downscale the image along depth, height and width
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            block_type (blocks.conv_blocks.BaseConvBlock): one the conv blocks inheriting from the BaseConvBlock class
            downsampling (blocks.conv.downsampling.Downscale): downsampling scheme
            downscale_last (bool): whether or not to do downsampling on the final output
            dropout (float): dropout added to the layer
            patch_size (int) : patch size for patch embedding
            channel_embedding (int) : number of channel for patch embedding
        '''
        super(ConvPatchEncoder, self).__init__()
        self.patch_embedding = torch.nn.Conv3d(input_shape[0],channel_embedding,kernel_size=patch_size,stride=patch_size)
        self.input_shape = (1,input_shape[0],input_shape[1],input_shape[2],input_shape[3])

        output_emb = conv3d_output_dim(input_dim=self.input_shape,num_kernels=channel_embedding,kernel_size=patch_size,stride=patch_size,padding=0,dilation=1)

        input_enc = (channel_embedding,output_emb[2],output_emb[3],output_emb[4])

        self.conv_encoder = ConvEncoder(input_shape=input_enc,
                                        num_channels_list=num_channels_list,
                                        kernel_size=kernel_size,
                                        downscale_factor=downscale_factor,
                                        activation=activation,
                                        normalization=normalization,
                                        block_type=block_type,
                                        downsampling=downsampling,
                                        downscale_last=downscale_last,
                                        dropout=dropout)

    def forward(self, x):
            '''
            Parameters:
            x (torch.Tensor): (N,C_in,D,H,W) input size

            Returns:
            x (Tuple[torch.Tensor, List[torch.Tensor]]): output and list of skip connections
            skip_connections (list) : list of skip connections output
            '''
            data = x
            x = self.patch_embedding(x)
            x, skip_connections = self.conv_encoder(x)
            skip_connections.insert(0,data)
            return x , skip_connections


    def compute_output_dimensions(self):
        '''
        computes the dimensions at the end of each convolutional block
        Returns:
            dimensions (List[Tuple]): dimension at the end of each convolutional block (first ones are skip connections while the last one is output of encoder)
        '''

        dim = [self.input_shape]
        dim_encodor = self.conv_encoder.compute_output_dimensions()
        return dim +dim_encodor




