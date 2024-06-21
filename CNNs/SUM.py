import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

from .kernels import convKernel
from .dwtHaar import dwtHaar_2d, idwtHaar_2d


#
class __threshold_learn__(nn.Module):
    def __init__(self, nts):
        super(__threshold_learn__, self).__init__()
        self.t = nn.Parameter( torch.zeros((nts,1,1)) )
        
    def forward(self, D, thr):
        t = self.t * thr
        return( F.leaky_relu(D - t,1e-3) - F.leaky_relu(-D - t,1e-2)  )
    
    
class downBlock(nn.Module):
    def __init__( self, pc, oc):
        super(downBlock, self).__init__()
        # Convolution layers
        self.c1 = nn.Conv2d(pc, oc,  kernel_size = (3, 3), bias=False, stride=1, padding=1, padding_mode='replicate')
        self.s1 = __threshold_learn__( oc )
        self.c2 = nn.Conv2d(oc, oc,  kernel_size = (3, 3), bias=False, stride=1, padding=1, padding_mode='replicate')
        self.s2 = __threshold_learn__( oc )
        
        self.dwt= dwtHaar_2d()

    def forward(self,x, thr):
        c1 = self.s1(self.c1(x ), thr)
        c2 = self.s2(self.c2(c1), thr)
        return( self.dwt( c2 ), c2 )

    
class upBlock(nn.Module):
    def __init__( self, pc, oc):
        super(upBlock, self).__init__()
        self.c1 =  nn.Conv2d(oc, oc,  kernel_size = (3, 3), bias=False, stride=1, padding=1, padding_mode='replicate')
        self.c12 =  nn.Conv2d(oc, oc,  kernel_size = (3, 3), bias=False, stride=1, padding=1, padding_mode='replicate')
        self.s1 = __threshold_learn__( oc )
        self.c2 =  nn.Conv2d(oc, pc,  kernel_size = (3, 3), bias=False, stride=1, padding=1, padding_mode='replicate')
        self.s2 = __threshold_learn__( pc )
        
        self.idwt = idwtHaar_2d()
    
    def forward(self,L, H, f, thr):
        y = self.idwt(L, H)
        c1 = self.s1 (self.c1(y )+self.c12(f), thr)
        c2 = self.s2( self.c2(c1), thr)
        return( c2 )
    
class outputLayer(nn.Module):
    def __init__( self):
        super(outputLayer, self).__init__()
        self.out = nn.Conv2d( 1, 1, 1, stride=1, padding=0, bias=False )
    def forward(self, x):
        return( torch.clamp(self.out( x ),0,1))
    
    
class deepestLayer(nn.Module):
    def __init__( self, pc, oc ):
        super(deepestLayer, self).__init__()
        self.c1 = nn.Conv2d(pc, oc,  kernel_size = (3, 3), bias=False, stride=1, padding=1, padding_mode='replicate')
        self.c2 = nn.Conv2d(oc, pc,  kernel_size = (3, 3), bias=False, stride=1, padding=1, padding_mode='replicate')
        self.s1 = __threshold_learn__( oc )
        self.s2 = __threshold_learn__( pc )
        
    def forward(self, x, thr):
        c1 = self.s1(self.c1(x ), thr ) 
        c2 = self.s2(self.c2(c1), thr ) 
        return( c2 )
        
class SUM_2d(nn.Module):
    def __init__( self, in_channels=1, depth=1, wf=4):
        super(SUM_2d, self).__init__()
        self.depth = depth
        
        # Encoding-decoding path
        self.enc = nn.ModuleList()
        self.dec = nn.ModuleList()
        pc = in_channels 
        for i in range(depth):
            oc = wf*2**i
            # Weight for encoding and  decoding blocks
            self.enc.append(downBlock(pc, oc))
            self.dec.append(upBlock(pc, oc))
            pc = oc

        self.ol = outputLayer()
        
        # Deepest layer
        oc = wf*2**(i+1)
        self. dl = deepestLayer(pc, oc)

    def forward(self, x, thr=1.0):
        # Encoder network 
        L = x
        
        H_list = []
        F_list = []
        for i, enc in enumerate(self.enc):
            LH, F = enc(L, thr)
            L, H = LH
            H_list.append( H )
            F_list.append( F )
        
        # Deepest layer
        L = self.dl(L, thr)
        
        # Decoder pass
        for i, dec in reversed(list(enumerate(self.dec))):
            L  = dec(L, H_list[i], F_list[i], thr)
        
        # Return values
        return(self.ol( L ), L)