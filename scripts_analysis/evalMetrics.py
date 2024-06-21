import numpy as np
from scipy.fftpack import fft2, fftshift
import torch
import torch.nn.functional as F

def impRes(cnn, isRef):
    # Defining an unit impulse
    imp = torch.zeros([1,1,128,128]).cuda()
    imp[0,0,64,64] = 1
    impnp = imp[0,0,:,:].detach().cpu().numpy() 
    if isRef:
        
        return(impnp, impnp)
    else:
        # Feeding the impulse to the CNN
        with torch.no_grad():
            p=128
            resp = cnn(
                F.pad(imp,(p,p,p,p)), thr=0 
                )[1][:,:,p:-p,p:-p] 
        respnp = resp[0,0,:,:].detach().cpu().numpy()
        return(respnp, impnp)

def diceScore(groundTruth, estimate):
    estimateBin = estimate>0.5
    overlap = np.sum(estimateBin*groundTruth)
    uni = np.sum(estimateBin) + np.sum(groundTruth)
    return(2*overlap/uni)

def Jaccard(groundTruth, estimate):
    estimateBin = estimate>0.5
    groundTruthBin = groundTruth>0.5
    
    inter = np.sum( np.logical_and(estimateBin, groundTruth) )
    union = np.sum( np.logical_or(estimateBin, groundTruth) )
    return( inter/union )


def td_spread(respnp, impnp):
    x = respnp
    
    xs = x.shape
    cx, cy = np.where(impnp>0)
    
    xs = x.shape
    Y, X = np.meshgrid( np.arange(xs[0]), np.arange(xs[1]) )
    distance = np.sqrt( (cx - X)**2 + (cy - Y)**2 )
    xpn = np.abs(x) /np.sum( np.abs(x) )

    return( np.sum( xpn*distance ) )


def fd_spread(respnp):
    x = np.absolute(fftshift(fft2( respnp  ) ) )
    
    xs = x.shape
    cx = xs[0]/2
    cy = xs[1]/2
    
    # It is even
    if xs[0]//2 == 0 :
        cx+=0.5
    if xs[1]//2 == 0 :
        cy+=0.5
    
    xs = x.shape
    Y, X = np.meshgrid( np.arange(xs[0]), np.arange(xs[1]) )
    distance = np.sqrt( (cx - X)**2 + (cy - Y)**2 )
    xpn = np.abs(x) /np.sum( np.abs(x) )

    return( np.sum( xpn*distance ) )


def fd_flatness(respnp):
    
    x = np.absolute(fftshift(fft2( respnp  ) ))
    frobNorm = np.sqrt(np.sum(x**2))
    
    Gx = np.asarray([[1,2,1], [0,0,0], [-1,-2,-1]])
        
    Gy = np.asarray([[1,0,-1], [2,0,-2], [1,0,-1]])
    from scipy.signal import convolve
    Dx = convolve(x, Gx, mode='valid')
    Dy = convolve(x, Gy, mode='valid')
    
    return( np.sum( np.sqrt( Dx**2 + Dy**2 ))/frobNorm  )

def evalAndComputeTVSP(cnn, isRef=False):
    respnp, impnp = impRes(cnn, isRef)
    tds = td_spread(respnp, impnp)
    
    fds = fd_spread(respnp)
    fdv = fd_flatness(respnp)
    
    return(tds, fdv, fds)