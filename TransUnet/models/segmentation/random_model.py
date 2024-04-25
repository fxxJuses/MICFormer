import torch

from segmentation.segmentation import SegmenationModel


class RandomModel(SegmenationModel):
    def __init__(self,
                 input_shape,
                 num_classes):
        '''
        Implementation of a random model

        Parameters:
            input_shape (tuple): (C,D,H,W) of the input
            num_classes (int): number of classes in the segmentation
        '''
        super(RandomModel, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

    def forward(self, x):
        '''
        Forward pass of the model

        Parameters:
            x (torch.Tensor): input tensor of shape (B,C,D,H,W)

        Returns:
            torch.Tensor: output tensor of shape (B,C,D,H,W)
        '''
        return torch.rand(x.shape[0], self.num_classes, *self.input_shape[1:])