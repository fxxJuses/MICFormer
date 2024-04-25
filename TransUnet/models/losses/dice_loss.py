import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1e-7,log_loss=False):
        '''
        Dice loss implementation for multiclasses segmentation purposes

        https://github.com/BloodAxe/pytorch-toolbelt/blob/develop/pytorch_toolbelt/losses/functional.py
        '''

        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.log_loss = log_loss

    def forward(self,pred,target):
        '''

        Parameters
        ----------
        pred (torch.Tensor) : tensors which contains prediction (N,NC,D,H,W)
        target (torch.Tensor): tensor which contains actual labels (N,D,H,W)
        N = batch size , NC = number of classes, D= depth , H = height, W = width

        Returns

        dice score loss (float): loss value for dice score
        -------
        '''

        # get dimensions of samples
        batch_size = pred.shape[0]
        n_class = pred.shape[1]

        # we get one hot encoding of target prediction
        target = nn.functional.one_hot(target, n_class)
        # we permute to get same shape as prediction tensor
        target = torch.permute(target, (0,4,1,2,3))

        # check if pred size and target size are a match
        assert pred.shape == target.shape

        # logits to probability for prediction
        pred = nn.functional.softmax(pred,dim=1)

        # we switch to flatten version of vectors
        pred = pred.view(batch_size, n_class, -1)
        target = target.view(batch_size,n_class,-1)

        # calculate multi-class dice loss
        inter = torch.sum(pred*target)
        card  = torch.sum(pred+target)
        dice_score = 2*(inter+self.smooth)/(card+self.smooth)
        loss =  1 - dice_score if not self.log_loss else -torch.log(dice_score)
        return loss
