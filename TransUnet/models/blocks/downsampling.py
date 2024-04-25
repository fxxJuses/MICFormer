import torch.nn as nn

class Downscale(nn.Module):
    def __init__(self, downscale_factor):
        '''
        Downsample in 3d
        Parameters:
            downscale_factor (int): factor by which to downscale the tensor
        '''
        super(Downscale, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        '''
        Parameters:
            x (torch.Tensor): (N, C, D, H, W)

        Returns:
            x (torch.Tensor): (N, C, D/scale, H/scale, W/scale)
        '''
        x = self.pool(x)
        return x


class MaxPool3dDownscale(Downscale):
    def __init__(self, downscale_factor):
        '''
        Downsample with maxpooling in 3d
        Parameters:
            downscale_factor (int): factor by which to downscale the tensor
        '''
        super(MaxPool3dDownscale, self).__init__(downscale_factor)
        self.pool = nn.MaxPool3d(self.downscale_factor, stride=self.downscale_factor)
    

class AvgPool3dDownscale(Downscale):
    def __init__(self, downscale_factor):
        '''
        Downsample with average pooling in 3d
        Parameters:
            downscale_factor (int): factor by which to downscale the tensor
        '''
        super(AvgPool3dDownscale, self).__init__(downscale_factor)
        self.pool = nn.AvgPool3d(self.downscale_factor, stride=self.downscale_factor)
