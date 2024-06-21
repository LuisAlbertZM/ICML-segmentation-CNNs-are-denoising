import h5py
import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter as gf

class dataLoaderSeg(data.Dataset):
    def __init__(self, dataset, iDs, enableRot=False, noise=False, blur=False):
        
        self.iDs = iDs
        self.noise = noise
        self.blur = blur
        
        # Getting image dimensions
        with h5py.File(dataset, 'r' ) as f:
            img_wiTum = f["%s_wiTumor_img"%(iDs[0])][:]
            sx, sy = img_wiTum.shape
        
        self.imgs = np.zeros((len(iDs)*2, 1, sx, sy))
        self.msks = np.zeros((len(iDs)*2, 1, sx, sy))
        self.enableRot = enableRot
        
        
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
        img = self.imgs[indx,0,:,:]
        msk = self.msks[indx,0,:,:]
        if self.enableRot:
            if np.random.rand(1)>0.5:
                img = np.fliplr(img)
                msk = np.fliplr(msk)
            if np.random.rand(1)>0.5:
                img = np.rot90(img, k=np.random.randint(0,3))
                msk = np.rot90(msk, k=np.random.randint(0,3))
        if self.noise:
            if np.random.rand(1)>0.5:
                img = img + 25.5*np.random.normal(np.zeros_like(img), np.ones_like(img))
        if self.blur:
            if np.random.rand(1)>0.5:
                img = gf(img, sigma=1.0)
                
        return([
            torch.from_numpy(img.copy()).unsqueeze(0).to(torch.float).cuda()/255,
            torch.from_numpy(msk.copy()).unsqueeze(0).to(torch.float).cuda()])

    def __len__(self):
        return( len(self.iDs)*2 )
    
    
    
class dataLoaderDenoising(data.Dataset):
    def __init__(self, dataset, iDs):
        
        self.iDs = iDs
        # Getting image dimensions
        with h5py.File(dataset, 'r' ) as f:
            img_wiTum = f["%s_wiTumor_img"%(iDs[0])][:]
            sx, sy = img_wiTum.shape
        
        self.imgs = np.zeros((len(iDs)*2, 1, sx, sy))
        
        
        with h5py.File(dataset, 'r' ) as f:
            for i , iD in enumerate(list(iDs)):
                img_wiTum = f["%s_wiTumor_img"%(iD)]
                img_noTum = f["%s_noTumor_img"%(iD)]
                
                lowIndx = i*2
                higIndx = i*2+1
                
                self.imgs[lowIndx, :] = img_noTum
                self.imgs[higIndx, :] = img_wiTum
        

    def __getitem__(self, indx):
                
        img = self.imgs[indx,0,:,:].astype(np.float)/255
        if np.random.rand(1)>0.5:
            img = np.fliplr(img.copy()).copy()
        if np.random.rand(1)>0.5:
            img = np.rot90(img.copy(), k=np.random.randint(0,3)).copy()

        imgNoisy = np.random.normal(img.copy(), 0.1*np.ones_like(img)).copy()

        return([
            torch.from_numpy(imgNoisy).unsqueeze(0).to(torch.float).cuda(),
            torch.from_numpy(img).unsqueeze(0).to(torch.float).cuda()])

    def __len__(self):
        return( len(self.iDs)*2 )