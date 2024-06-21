import torch
import torch.nn.functional as F

def soft_threshold(x,b):
    return(F.relu(x - b )-F.relu(-x - b ))

def soft_sat(x,b):
    return(x - (F.relu(x - b )-F.relu(-x - b )) )