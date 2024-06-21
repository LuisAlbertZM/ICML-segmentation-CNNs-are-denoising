import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Sequential

# NETWORKS
################################
# UNET ARCHITECTURE ############
################################
class __unet_conv_block__(nn.Module):
    def __init__(self, indf, ondf):
        super(__unet_conv_block__, self).__init__()
        self.cblock1 = Sequential(
            nn.Conv2d(indf, ondf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ondf),
            nn.LeakyReLU(inplace=True,negative_slope=1e-2))
        self.cblock2 = Sequential(
            nn.Conv2d(ondf, ondf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ondf),
            nn.LeakyReLU(inplace=True,negative_slope=1e-2),
            nn.ReflectionPad2d(padding =(0,1,0,1)),
            nn.MaxPool2d(kernel_size=2))
    def forward(self, x):
        conv = self.cblock1(x)
        return(self.cblock2(conv), conv)

class __unet_up_block__(nn.Module):
    def __init__(self, indf, ondf, kernel_size=3, padding=1):
        super(__unet_up_block__, self).__init__()
        self.up = nn.Sequential( 
            nn.ConvTranspose2d(ondf, ondf, kernel_size = 1, stride=2, output_padding=1),
            nn.BatchNorm2d(ondf),
            nn.LeakyReLU(inplace=True,negative_slope=1e-2))
        self.reduce = nn.Sequential(
            nn.Conv2d(ondf*2, ondf, kernel_size=1, padding=0),
            nn.BatchNorm2d(ondf),
            nn.LeakyReLU(inplace=True,negative_slope=1e-2))
        self.cblock = nn.Sequential(
            nn.Conv2d(ondf  , indf, kernel_size=3, padding=1),
            nn.BatchNorm2d(indf),
            nn.LeakyReLU(inplace=True,negative_slope=1e-2))
    def forward(self, x, bridge):
        conc = torch.cat([self.up(x),bridge],1)
        red = self.reduce(conc)
        conv = self.cblock(red)
        return  conv

class deepestLayer(nn.Module):
    def __init__( self, pc ):
        super(deepestLayer, self).__init__()
        self.cb = nn.Sequential(
            nn.Conv2d(pc  , pc*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(pc*2),
            nn.LeakyReLU(inplace=True,negative_slope=1e-2),
            nn.Conv2d(pc*2  , pc, kernel_size=3, padding=1),
            nn.BatchNorm2d(pc),
            nn.LeakyReLU(inplace=True,negative_slope=1e-2),
        )
        
    def forward(self, x ):
        return( self.cb(x) )
    
class outputLayer(nn.Module):
    def __init__(self, ic, oc):
        super(outputLayer, self).__init__()
        self.out = nn.Conv2d( ic, oc, 1, stride=1, padding=0, bias=False )
    def forward(self, x):
        return( torch.clamp(self.out( x ),0,1))
    

class unet_2d(nn.Module):
    def __init__( self, inout_chans=1, depth=5, wf=16):
        super(unet_2d, self).__init__()
        self.depth = depth
        #  Begining architecture
        pc = inout_chans
        
        self.down_path = nn.ModuleList()
        self.up_path   = nn.ModuleList()
        for i in range(depth):
            oc = wf*2**i
            self.down_path.append( __unet_conv_block__(pc, oc) )
            self.up_path.append( __unet_up_block__(pc, oc) )
            pc = oc 

        self.dep = deepestLayer(oc)
        self.ol = outputLayer(inout_chans, inout_chans)
        
    def forward(self, x):
        blocks = []
        bridges = []
        L = x
        for i, down in enumerate(self.down_path):
            L, bridge = down(L)
            bridges.append(bridge)
            
        L = self.dep(L)
        
        for i, up in reversed(list(enumerate(self.up_path))):
            L = up(L, bridges[i])
            
        return( self.ol(L), L )