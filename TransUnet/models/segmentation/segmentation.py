import torch.nn as nn
import torch

class SegmentationModel(nn.Module):
    def __init__(self):
        '''
        base segmentation model
        '''
        super(SegmentationModel, self).__init__()

        #input is (N,C,D,H,W)
        self.predict_softmax = nn.Softmax(dim=1)

    def predict_proba(self, x):
        '''
        Return the probabilities from the forward ouput
        Parameters:
            x (torch.Tensor): (N, C_in, D_in, H_in, W_in)
        Parameters:
            x (torch.Tensor): (N, num_classes, D_in, H_in, W_in)
        '''
        x = self.forward(x)
        x = self.predict_softmax(x)
        return x
    
    def predict(self, x):
        '''
        Returns the predicted label from the forward ouput
        Parameters:
            x (torch.Tensor): (N, C_in, D_in, H_in, W_in)
        Parameters:
            x (torch.Tensor): (N, num_classes, D_in, H_in, W_in)
        '''
        x = self.forward(x)
        x = torch.argmax(x, 1, keepdim=True)
        return x




