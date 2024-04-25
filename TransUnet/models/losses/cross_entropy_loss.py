import torch.nn as nn


class CrossEntropy(nn.Module):
    '''

    '''
    def __init__(self,label_smoothing= 0.0):
        super(CrossEntropy, self).__init__()
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self,pred,target):
        '''

        Parameters
        ----------
        pred (torch.Tensor) : tensors which contains prediction (N,D,H,W)
        target (torch.Tensor): tensor which contains actual labels (N,D,H,W)

        Returns
        -------
        loss (float): CE loss value
        '''

        return self.criterion(pred,target)