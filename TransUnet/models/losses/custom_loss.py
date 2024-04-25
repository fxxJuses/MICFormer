import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self,list_loss:nn.ModuleList,list_pond: list):
        '''

        Parameters
        ----------
        list_loss : list of different loss to mesure
        list_pond : list of ponderations of different loss in list loss
        '''
        super(CustomLoss, self).__init__()
        self.list_loss = list_loss
        self.list_pond = list_pond

    def forward(self,predict,target):
        cumul_loss = 0
        for i,loss in enumerate(self.list_loss):
            cumul_loss += self.list_pond[i] * loss(predict,target)
        return cumul_loss
        
