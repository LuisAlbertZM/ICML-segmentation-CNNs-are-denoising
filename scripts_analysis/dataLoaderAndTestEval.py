import h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils import data
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.utils import data


class dataDummy(data.Dataset):
    def __init__(self, dataset, iDs, blur=False, noise=False, rotate=False, angle=0):
        
        self.blur=blur
        self.noise = noise
        self.rotate = rotate
        self.angle = angle
        self.iDs = iDs
        
        # Getting image dimensions
        with h5py.File(dataset, 'r' ) as f:
            img_wiTum = f["%s_wiTumor_img"%(iDs[0])][:]
            sx, sy = img_wiTum.shape
        
        self.imgs = np.zeros((len(iDs)*2, 1, sx, sy))
        self.msks = np.zeros((len(iDs)*2, 1, sx, sy))
        
        with h5py.File(dataset, 'r' ) as f:
            for i , iD in enumerate(list(iDs)):
                img_wiTum = f["%s_wiTumor_img"%(iD)]
                msk_wiTum = f["%s_wiTumor_seg"%(iD)]
                img_noTum = f["%s_noTumor_img"%(iD)]
                msk_noTum = f["%s_noTumor_seg"%(iD)]
                
                lowIndx = i*2
                higIndx = i*2+1
                
                self.imgs[lowIndx, :] = img_noTum
                self.imgs[higIndx, :] = img_wiTum
                self.msks[lowIndx, :] = msk_noTum
                self.msks[higIndx, :] = msk_wiTum
        

    def __getitem__(self, indx):
        
        from scipy.ndimage import gaussian_filter as gf
        img = self.imgs[indx,:]
        msk = self.msks[indx,:]
        
        filt = img
        if self.noise:
            filt = filt + 25.5*np.random.normal(np.zeros_like(filt), np.ones_like(filt))
        if self.blur:
            filt = gf(filt, sigma=1.0)
        if self.rotate:
            from scipy import ndimage
            filt = ndimage.rotate(filt, angle=self.angle, reshape=False, order=2, mode='nearest', axes=(1,2))
            msk = ndimage.rotate(msk, angle=self.angle, reshape=False, order=0, mode='nearest', axes=(1,2)) 
            
        return([
            torch.clamp( torch.from_numpy(filt).to(torch.float).cuda()/255, 0,1),
            torch.from_numpy(msk).to(torch.float).cuda()])

    def __len__(self):
        return( len(self.iDs)*2 )

def evalTestSet(cnn, dataLoader):
    evalTensor = np.zeros( (len(dataLoader),128,128))
    with torch.no_grad():
        for i, (im, _) in enumerate(dataLoader):
            ev, delta = cnn( im.unsqueeze(0) )[0:2]
            evalTensor[i,:,:] = ev.cpu().detach().numpy()[0,0]
    return(evalTensor)