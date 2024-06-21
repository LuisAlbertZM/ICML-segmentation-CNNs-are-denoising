import h5py
import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader


def valLoss(val_data, cnn, lossFunc):
    loss=0
    dldr = DataLoader(val_data, batch_size=1, shuffle=False)
    for (inp,targ) in dldr:
        with torch.no_grad():
            loss += lossFunc(cnn, inp, targ).item()
    return(loss/val_data.__len__())
     
def trainingLoop(
    cnn,
    train_data,
    valid_data,
    epo, bs,
    trainLossFunc,
    validLossFunc,
    opt,
    ilr, modlName, printLog=False, logName=""):
    # Head of the log
    if printLog:
        sourceFile = open(logName, 'w')
        print('Epoch ,LR , Loss', file = sourceFile)
        sourceFile.close()
        
    # Main loop 
    lr = ilr 
    loss_prev = np.Inf
    
    for e in range(epo):
        #lgen=0
        dldr = DataLoader(train_data, batch_size=bs, shuffle=True)
        for j, (inp, targ) in enumerate(dldr):
            opt.zero_grad()
            loss =  trainLossFunc(cnn, inp, targ)
            loss.backward()
            
            #torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            torch.nn.utils.clip_grad_norm_(cnn.parameters(), 1e-3) # or some other value
            opt.step()
        #print(e)
        
            

        # Display learning rate
        lrd = lr
        
        # Adjusting the linearly decaying learning rate
        lr = ilr - (ilr/epo)*e
        opt.param_groups[0]['lr'] = lr
        
        # Computing training and validation losses
        tl = valLoss(train_data, cnn, trainLossFunc)
        vl = valLoss(valid_data, cnn, validLossFunc)
        
        ## Save model
        #if vl < loss_prev:
        loss_prev = vl
        torch.save(cnn.state_dict(), modlName)
        sav = 1
        #else:
        #    sav = 0
        
        # Logging
        print("Epoch: %s ,LR: %1.5f, Train Loss: %1.5f, Val Loss: %1.5f, Saved: %s"%(str(e), lrd, tl, vl, str(sav) ))
        if printLog:
            sourceFile = open(logName, 'a')
            print("%s , %1.5f, %1.5f, %1.5f, %s"%(str(e), lrd, tl, vl, str(sav) ), file = sourceFile)
            sourceFile.close()
        

        
def trainingLoopFullScan(
    cnn,
    train_data,
    valid_data,
    epo, bs,
    lossFunc,
    opt,
    ilr, modlName, printLog=False, logName=""):
    # Head of the log
    if printLog:
        sourceFile = open(logName, 'w')
        print('Epoch ,LR , Loss', file = sourceFile)
        sourceFile.close()
        
    # Main loop 
    lr = ilr 
    loss_gen_prev = np.Inf
    
    for e in range(epo):
        #lgen=0
        for sc in train_data:
            dldr = DataLoader(sc, batch_size=bs, shuffle=True)
            for j, (inp, targ) in enumerate(dldr):
                loss =  lossFunc(cnn, inp, targ)
                loss.backward()
                opt.step()
                opt.zero_grad()

        # Display learning rate
        lrd = lr

        # Adjusting the linearly decaying learning rate
        lr = ilr - (ilr/epo)*e
        opt.param_groups[0]['lr'] = lr

        # Computing training and validation losses
        tl = valLossFullScan(train_data, cnn, lossFunc)
        vl = valLossFullScan(valid_data, cnn, lossFunc)

        print("Epoch: %s ,LR: %1.5f, Train Loss: %1.5f, Val Loss: %1.5f"%(str(e), lrd, tl, vl ))

        # Print log to file
        if printLog:
            sourceFile = open(logName, 'a')
            print("%s , %1.5f, %1.5f, %1.5f"%(str(e), lrd, tl, vl ), file = sourceFile)
            sourceFile.close()

        # Save model
        torch.save(cnn.state_dict(), modlName)