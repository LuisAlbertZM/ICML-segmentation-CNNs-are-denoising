import numpy as np
import torch
import torch.nn as nn

    
    

# Adapted from
# https://github.com/hubutui/DiceLoss-PyTorch/blob/master/loss.py
class BinaryIoU(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryIoU, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = torch.clamp( predict.contiguous().view(predict.shape[0], -1), min=0, max=1)
        target = torch.clamp( target.contiguous().view(target.shape[0], -1) , min=0, max=1)

        num = torch.sum( predict * target)#, dim=1 )# + self.smooth
        den =  torch.sum(predict + target- predict * target)#, dim=1 )  

        loss = 1 - num / (den + self.smooth )

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))
            
            
            
