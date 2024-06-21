########################################################
# Author: Luis Albert Zavala Mondragon
# Organization: Eindhoven University of Technology
########################################################
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F


""" class: __upsample_zeropadding_2D__
    Description: Compact implementation of 2D upsampling via insertion 
    of zeros
    Constructor:
        * None
    Inputs:
        * x: Signal to be upsampled
    Outputs:
        * upsi: Upsampled signal
"""
class __upsample_zeropadding_2D__(nn.Module):

    def __init__(self):
        super(__upsample_zeropadding_2D__, self).__init__()
        
        k_np = np.asanyarray([1])
        self.k = nn.Parameter(data = torch.from_numpy(k_np),
            requires_grad=False).float().cuda().reshape((1,1,1,1))
        
    def forward(self, x):
        xs = x.shape
        x_p = x.view(xs[0]*xs[1], 1, xs[2], xs[3])
        
        up = F.conv_transpose2d(x_p, weight=self.k, stride=(2,2), dilation=1)
            
        if up.shape[2] < x.shape[2]*2:
            up = F.pad(input = up, pad = (0, 0, 0, 1), mode="replicate", value=0)
        if up.shape[3] < x.shape[3]*2:
            up = F.pad(input = up, pad = (0, 1, 0, 0), mode="replicate", value=0)
        
        us = up.shape
        upsi = up.view(xs[0], xs[1], us[2], us[3])
        return(upsi)
    
"""
    Description: Compact implementation of inverse 2D DWT with 
    Haar kernel
"""
class dwtHaar_2d(nn.Module):
    def __init__( self, undecimated=False, mode="replicate"):
        super(dwtHaar_2d, self).__init__()
        self.mode = mode
        if undecimated: self.stride=1
        else: self.stride=2
        k = 0.5
    
        # 2D Haar DWT
        self.WL = torch.tensor(
            [[[[ k,  k], [ k,  k]]]],
            requires_grad=False
            ).to(torch.float32).cuda().transpose(1,0)
        self.WH = torch.tensor(
            [[[[ k,  k], [-k, -k]],
              [[-k,  k], [-k,  k]],
              [[ k, -k], [-k,  k]]]],
            requires_grad=False
            ).to(torch.float32).cuda().transpose(1,0)


    def forward(self,x):
        # Loading parameters
        mode = self.mode
        stride = self.stride

        # Reshaping for easy convolutions
        #with torch.no_grad():
        # First wavelet transform, second, wavelet transform
        xlc = F.pad(x, (1,0,1,0), mode=mode, value=0)
        xps = xlc.shape
        xlcs = xlc.shape
        
        xlc = xlc.view(xlcs[0]*xlcs[1], 1, xlcs[2], xlcs[3])
        L = F.conv2d(xlc, self.WL, stride=stride)
        H = F.conv2d(xlc, self.WH, stride=stride)
            
        osil = [xlcs[0], xlcs[1]  , L.shape[2], L.shape[3]]
        osih = [xlcs[0], xlcs[1]*3, L.shape[2], L.shape[3]]
        return([L.view(*osil), H.view(*osih)])

"""
    Description: Compact implementation of inverse 2D DWT with 
    Haar kernel
"""
class idwtHaar_2d(nn.Module):
    def __init__( self,mode="replicate"):
        super(idwtHaar_2d, self).__init__()
        self.mode = mode
        self.upsample = __upsample_zeropadding_2D__()
        k = 0.5
        self.iWL = torch.tensor(
            [[[[ k,  k], [ k,  k]]]], 
            requires_grad=False).to(torch.float32).cuda()
        self.iWH = torch.tensor(
            [[[[-k, -k], [ k,  k]], 
              [[ k, -k], [ k, -k]],
              [[ k, -k], [-k,  k]]]],
              requires_grad=False).to(torch.float32).cuda()
        
    def forward(self, L, H):
        # Loading parameters
        mode=self.mode
        stride=1

        # Reshaping for easy convolutions
        #with torch.no_grad():
        # Upsampling by inserting zeros
        x =  self.upsample(torch.cat([L, H],axis=1))
        xp = F.pad(x,(0,1,0,1),mode=mode,value=0)
        L2, H2 = torch.split(xp, (L.shape[1], L.shape[1]*3), dim=1)

        lls = L2.shape
        # Inverse transform
        csl = [lls[0]*lls[1], 1, lls[2], lls[3]]
        csh = [lls[0]*lls[1], 3, lls[2], lls[3]]
        iL = F.conv2d(L2.reshape(*csl), self.iWL, stride=stride)
        iH = F.conv2d(H2.reshape(*csh), self.iWH, stride=stride)
        xiw = iL + iH
        ills = iL.shape
        
        co = [lls[0], lls[1], ills[2], ills[3]]
        return(xiw.reshape(*co))