import torch.nn as nn
import torch

class Conv3DDropoutActivation(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 activation=nn.ReLU,
                 dropout=0
                 ):
        '''
        3D convolution followed by dropout then non linear activation
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            dropout (float): dropout added to the layer
        '''
        super(Conv3DDropoutActivation, self).__init__()
        self.convolution = torch.nn.Conv3d(
                                    in_channels, 
                                    out_channels, 
                                    kernel_size, 
                                    stride=stride, 
                                    padding=padding, 
                                    dilation=dilation
                                )
        self.dropout = nn.Dropout3d(p=dropout)
        self.activation = activation()

    def forward(self, x):
        '''
        Parameters:
            x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
            x (torch.Tensor): (N,C_out,D,H,W) output size
        '''
        x = self.convolution(x)
        x = self.dropout(x)
        x = self.activation(x)
        return x
  

class Conv3DDropoutNormActivation(Conv3DDropoutActivation):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1,  
                 activation=nn.ReLU, 
                 normalization=nn.BatchNorm3d,
                 dropout=0,
                 ):
        '''
        3D convolution followed by dropout then normalization and then non linear activation
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            dropout (float): dropout added to the layer
        '''
    
        super(Conv3DDropoutNormActivation, self).__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation, 
            activation=activation,
            dropout=dropout
            )
        self.normalization = normalization(out_channels)

    def forward(self, x):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
        x (torch.Tensor): (N,C_out,D,H,W) output size
        '''
        x = self.convolution(x)
        x = self.dropout(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x
    

class BaseConvBlock(nn.Module):
    def __init__(
            self, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1,  
            activation=nn.ReLU, 
            normalization=nn.BatchNorm3d,
            dropout=0,
            ):
        super(BaseConvBlock, self).__init__()
        '''
        Initializes the self.baseblock function (either normalization or not)
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            dropout (float): dropout added to the layer
        '''

        if normalization is None:
            self.base_block = lambda in_channels: Conv3DDropoutActivation(
                                in_channels, 
                                out_channels, 
                                kernel_size, 
                                stride=stride, 
                                padding=padding, 
                                dilation=dilation,  
                                activation=activation,
                                dropout=dropout 
                            )
        else:
            self.base_block = lambda in_channels: Conv3DDropoutNormActivation(
                                in_channels, 
                                out_channels, 
                                kernel_size, 
                                stride=stride, 
                                padding=padding, 
                                dilation=dilation,  
                                activation=activation, 
                                normalization=normalization,
                                dropout=dropout 
                            )
            
    def forward(self, x):
        raise NotImplementedError
    

class SingleConvBlock(BaseConvBlock):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1,  
            activation=nn.ReLU, 
            normalization=nn.BatchNorm3d,
            dropout=0
            ):
        super(SingleConvBlock, self).__init__( 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,  
            activation=activation, 
            normalization=normalization,
            dropout=dropout,
        )
        '''
        Combines 2 basic convolutional blocks into one.
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            dropout (float): dropout added to the layer
        '''
            
        self.conv_block =  self.base_block(in_channels)

            
    def forward(self, x):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
        x (torch.Tensor): (N,C_out,D,H,W) output size
        '''
        x = self.conv_block(x)
        return x

    

class DoubleConvBlock(BaseConvBlock):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1,  
            activation=nn.ReLU, 
            normalization=nn.BatchNorm3d,
            dropout=0
            ):
        super(DoubleConvBlock, self).__init__( 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,  
            activation=activation, 
            normalization=normalization,
            dropout=dropout,
        )
        '''
        Combines 2 basic convolutional blocks into one.
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            dropout (float): dropout added to the layer
        '''
            
        self.conv_block_1 =  self.base_block(in_channels)
        self.conv_block_2 =  self.base_block(out_channels)

            
    def forward(self, x):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
        x (torch.Tensor): (N,C_out,D,H,W) output size
        '''
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x
    

class ResConvBlock(BaseConvBlock):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1,  
            activation=nn.ReLU, 
            normalization=nn.BatchNorm3d,
            dropout=0,
            ):
        super(ResConvBlock, self).__init__(
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,  
            activation=activation, 
            normalization=normalization,
            dropout=dropout
        )
        '''
        Combines 3 basic convolutional blocks into one with a residual connection.
        Inspired by https://arxiv.org/pdf/1706.00120.pdf
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            dropout (float): dropout added to the layer
        '''
            
        self.conv_block_1 =  self.base_block(in_channels)
        self.conv_block_2 =  self.base_block(out_channels)
        self.conv_block_3 =  self.base_block(out_channels)

            
    def forward(self, x):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
        x (torch.Tensor): (N,C_out,D,H,W) output size
        '''
        #first convolution
        x = self.conv_block_1(x)

        #skip connection
        x_skip = x

        #second convolution
        x = self.conv_block_2(x)

        #third convolution
        x = self.conv_block_3(x)

        #skip connection
        x = x + x_skip
        return x
    


class ResConvBlockUnetr(BaseConvBlock):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1,  
            activation=nn.ReLU, 
            normalization=nn.BatchNorm3d,
            dropout=0,
            ):
        super(ResConvBlockUnetr, self).__init__(
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,  
            activation=activation, 
            normalization=normalization,
            dropout=dropout
        )
        '''
        Combines 2 basic convolutional blocks into one with a residual connection.
        Inspired by https://arxiv.org/abs/2201.01266
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            dropout (float): dropout added to the layer
        '''
            
        self.conv_block_1 =  self.base_block(in_channels)
        self.conv_block_2 =  self.base_block(out_channels)
        self.skip_resize = nn.Conv3d(in_channels, out_channels, 1, stride=1)

            
    def forward(self, x):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
        x (torch.Tensor): (N,C_out,D,H,W) output size
        '''

        #skip connection
        x_skip = x

        #resise skip connection
        x_skip = self.skip_resize(x_skip)

        #second convolution
        x = self.conv_block_1(x)

        #third convolution
        x = self.conv_block_2(x)

        #skip connection
        x = x + x_skip
        
        return x


class BaseConvNextBlock(nn.Module):
    def __init__(
            self,
            channels,  
            kernel_size, 
            stride=1, 
            activation=nn.GELU, 
            normalization=nn.BatchNorm3d,
            dropout=0,
            up_factor=3,
            ):
        super(BaseConvNextBlock, self).__init__()
        '''
        Base Block inspired by the ConvNext paper.
        Inspired by https://arxiv.org/abs/2201.03545
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            dropout (float): dropout added to the layer
        '''
        #depth convolution
        self.depth_conv = nn.Conv3d(channels, channels, kernel_size, stride=stride, padding='same', groups=channels)

        #dropout
        self.dropout = nn.Dropout(p=dropout)

        #normalization
        self.normalization = normalization(channels)

        #first conv
        self.conv1 = nn.Conv3d(channels, up_factor*channels, 1, stride=1)

        #non linearity
        self.activation = activation()

        #second conv
        self.conv2 = nn.Conv3d(up_factor*channels, channels, 1, stride=1)
        

            
    def forward(self, x):
        '''
        Parameters:
        x (torch.Tensor): (N,C,D,H,W) input size

        Returns:
        x (torch.Tensor): (N,C,D,H,W) output size
        '''

        #skip connection
        x_skip = x

        #depth convolution
        x = self.depth_conv(x)

        #dropout
        x = self.dropout(x)

        #normalize
        x = self.normalization(x)

        #conv 1
        x = self.conv1(x)

        #activation
        x = self.activation(x)

        #conv 2
        x = self.conv2(x)

        #skip connection
        x = x + x_skip
        
        return x



class ConvNextBLock(nn.Module):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1,  
            activation=nn.GELU, 
            normalization=nn.BatchNorm3d,
            dropout=0,
            up_factor=3,
            ):
        super(ConvNextBLock, self).__init__()
        '''
        Block inspired by the ConvNext paper.
        Inspired by https://arxiv.org/abs/2201.03545
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            dropout (float): dropout added to the layer
        '''

        #resize entry
        self.resize = nn.Conv3d(in_channels, out_channels, 1, stride=1)
        
        self.convnext = BaseConvNextBlock(
                            out_channels,  
                            kernel_size, 
                            stride=stride, 
                            activation=activation, 
                            normalization=normalization,
                            dropout=dropout,
                            up_factor=up_factor,
                        )
        

            
    def forward(self, x):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
        x (torch.Tensor): (N,C_out,D,H,W) output size
        '''

        #resize x 
        x = self.resize(x)

        #convnext block
        x = self.convnext(x)
        
        return x
    

class DoubleConvNextBLock(nn.Module):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1,  
            activation=nn.GELU, 
            normalization=nn.BatchNorm3d,
            dropout=0,
            up_factor=3,
            ):
        super(DoubleConvNextBLock, self).__init__()
        '''
        Block inspired by the ConvNext paper.
        Inspired by https://arxiv.org/abs/2201.03545
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            dropout (float): dropout added to the layer
        '''

        #resize entry
        self.resize = nn.Conv3d(in_channels, out_channels, 1, stride=1)
        
        self.convnext1 = BaseConvNextBlock(
                            out_channels,  
                            kernel_size, 
                            stride=stride, 
                            activation=activation, 
                            normalization=normalization,
                            dropout=dropout,
                            up_factor=up_factor,
                        )
        
        self.convnext2 = BaseConvNextBlock(
                            out_channels,  
                            kernel_size, 
                            stride=stride, 
                            activation=activation, 
                            normalization=normalization,
                            dropout=dropout,
                            up_factor=up_factor,
                        )
        

            
    def forward(self, x):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
        x (torch.Tensor): (N,C_out,D,H,W) output size
        '''

        #resize x 
        x = self.resize(x)

        #convnext block
        x = self.convnext1(x)

        #convnext block
        x = self.convnext2(x)
        
        return x