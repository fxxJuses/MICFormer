import torch
import torch.nn as nn
from models.blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock

class ConvSkipBloc(nn.Module):
    def __init__(self,
                 num_channels_list,
                 kernel_size,
                 activation=nn.ReLU,
                 normalization=nn.BatchNorm3d,
                 block_type=DoubleConvBlock,
                 dropout=0,
                 skip_leak=False,
                 ):
        '''
        Parameters:
            num_channels_list (int): list of number of channels in each block
            kernel_size (int or tuple): Size of the convolving kernel
            activation (def None -> torch.nn.Module): non linear activation used by the block
            dropout (float): dropout added to the layer
        '''
        super(ConvSkipBloc, self).__init__()
        self.skip_leak = skip_leak

        block_instanciator = lambda in_channels, out_channels: block_type(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            activation=activation,
            normalization=normalization,
            dropout=dropout,
        )
        self.modif_skip = nn.ModuleList()
        for channel in num_channels_list:
            self.modif_skip.append(block_instanciator(channel,channel))

    def forward(self, skips):
        '''
        Parameters:
            skips (list[torch.Tensor]): list of skip connection [(N,C_in,D,H,W)]

        Returns:
           skips (list[torch.Tensor]): list of modified skip connection (N,C_out,D,H,W) output size
        '''

        result = []

        for i,skip in enumerate(skips):
            new_skip = self.modif_skip[i](skip)
            if self.skip_leak:
                new_skip = new_skip +skip
            result.append(new_skip)
        return result