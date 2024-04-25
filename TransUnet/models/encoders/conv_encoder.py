import torch
import torch.nn as nn

#personal modules
from models.blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from models.blocks.downsampling import MaxPool3dDownscale, AvgPool3dDownscale
from utils.conv_utils import conv3d_output_dim

class ConvEncoder(nn.Module):
    def __init__(self,
                 input_shape,
                 num_channels_list,
                 kernel_size=3,
                 downscale_factor=2,
                 activation=nn.ReLU, 
                 normalization=nn.BatchNorm3d,
                 block_type=DoubleConvBlock,
                 downsampling=MaxPool3dDownscale,
                 downscale_last=False,
                 dropout=0,
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
        '''
        super(ConvEncoder, self).__init__()

        assert kernel_size % 2 == 1, "kernel size should be an odd number (standard practice)"

        self.num_channels_list = num_channels_list
        self.num_blocks=len(num_channels_list)
        self.input_shape = input_shape

        #conv parameters for same convolution
        self.kernel_size = kernel_size
        self.padding = kernel_size//2  

        #downscaling parameters
        self.downscale_factor = downscale_factor
        self.downscale_last = downscale_last

        #function that instanciates each block
        self.block_instanciator = lambda in_channels, out_channels: block_type(
                                                            in_channels, 
                                                            out_channels, 
                                                            kernel_size,  
                                                            padding=kernel_size//2,   
                                                            activation=activation, 
                                                            normalization=normalization,
                                                            dropout=dropout,
                                                        )
        
            
        #instanciate the encoding conv_layers (conv_blocks + downscaling_layers)
        self.conv_blocks = nn.ModuleList()
        self.downscaling_layers = nn.ModuleList()

        c_in = input_shape[0]
        for i, c_out in enumerate(self.num_channels_list):
            self.conv_blocks.append(self.block_instanciator(c_in, c_out))

            #number of input channels in the next block corresponds to output
            c_in = c_out
            if i < self.num_blocks - 1 or self.downscale_last:
                self.downscaling_layers.append(downsampling(self.downscale_factor))
        
        

    def forward(self, x):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
        x (Tuple[torch.Tensor, List[torch.Tensor]]): output and list of skip connections
        '''
        #we want to save the skip connections
        skip_connections = []

        #iterate over the number of blocks
        for i in range(self.num_blocks):

            #pass through convolutional block
            x = self.conv_blocks[i](x)

            #downscale unless last conv block
            if i < self.num_blocks - 1 or self.downscale_last:
                #save skip connection
                skip_connections.append(x)
                
                #downscale
                x = self.downscaling_layers[i](x)

        return x, skip_connections
    
    def compute_output_dimensions(self):
        '''
        computes the dimensions at the end of each convolutional block
        Returns:
            dimensions (List[Tuple]): dimension at the end of each convolutional block (first ones are skip connections while the last one is output of encoder)
        '''
        #output dimension at each 
        dimensions=[]
        dim = tuple([1] + list(self.input_shape))
        for i, c_out in enumerate(self.num_channels_list):
            dim = conv3d_output_dim(dim, c_out, self.kernel_size, 1, self.padding, 1)
            dimensions.append(dim)
            dim = conv3d_output_dim(dim, c_out, self.downscale_factor, self.downscale_factor, 0, 1)

        if self.downscale_last:
            dimensions.append(dim)

        return dimensions



