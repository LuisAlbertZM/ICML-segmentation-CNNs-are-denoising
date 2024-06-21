import numpy as np
import torch
from .evalMetrics import Jaccard, diceScore
from .dataLoaderAndTestEval import dataDummy, evalTestSet 

def evalAndComputeDiceJaccard(cnn, testSet):
    groundTruth = testSet.msks[:,0,:,:]
    seg = evalTestSet(cnn, testSet)
    ds =  diceScore(groundTruth, seg)
    IoU = Jaccard(groundTruth, seg)
    
    return(ds, IoU)



def generateTableResults(models, modelNames, dataset, display=False):
    m = models
    mn = modelNames
    
    m_dice = []
    m_IoU = []
    for i, m in enumerate(models):
        # Evaluating all the results 
        m0_dice, m0_IoU = evalAndComputeDiceJaccard(m, dataset)
        m_dice.append(m0_dice)
        m_IoU.append(m0_IoU)
    
    if display:
        print("Metric "+ "".join("& %s "%(m) for m in modelNames) +"\\\\")
        print("Dice Score "+ "".join("& %1.3f "%(m) for m in m_dice) +"\\\\")
        print("IoU "+ "".join("& %1.3f "%(m) for m in m_IoU) +"\\\\")
    return(np.asarray(m_dice), np.asarray(m_IoU) )


def table1DiceIoU(SUM_2d, unet_2d, unetNoBN_2d, SUMparams, UNetParams,  uNetNoBN, test_data, no_runs):
    modelNames = ["U-Net", "U-Net$_\mathrm{NoBN}$", "SUM$_\mathrm{Seg}$", "SUM$_\mathrm{Rec}$"]

    SUMrec = SUM_2d(*SUMparams).float().cuda()
    SUMseg = SUM_2d(*SUMparams).float().cuda()
    uNet = unet_2d(*UNetParams).float().cuda()
    uNetNoBN = unetNoBN_2d(*UNetParams).float().cuda()
    

    dic_runs = np.zeros((5,4))
    iou_runs = np.zeros((5,4))
    # Iterating through all the runs
    for r in np.arange(no_runs):

        # Loading weights for each run
        SUMseg.load_state_dict(torch.load( "./CNN_weights/SUM_SEG_run%s.pyt"%r))
        SUMseg.eval();
        SUMrec.load_state_dict(torch.load( "./CNN_weights/SUM_REC_run%s.pyt"%r))
        SUMrec.eval();
        uNet.load_state_dict(torch.load( "./CNN_weights/unet_run%s.pyt"%r))
        uNet.eval();
        uNetNoBN.load_state_dict(torch.load( "./CNN_weights/unet_NoBN_run%s.pyt"%r))
        uNetNoBN.eval();

        models = [uNet, uNetNoBN, SUMseg, SUMrec]

        # Computing Dice score and IoU
        dic, iou = generateTableResults(models, modelNames, test_data, display=False)
        dic_runs[r,:] = dic
        iou_runs[r,:] = iou

    # Computing mean and standard deviation for each of the runs
    mea_dic_runs = np.mean(dic_runs, axis=0).tolist()
    std_dic_runs = np.std(dic_runs, axis=0).tolist()
    mea_iou_runs = np.mean(iou_runs, axis=0).tolist()
    std_iou_runs = np.std(iou_runs, axis=0).tolist()

    # Displaying table
    print("Metric "+ "".join("& %s "%(m) for m in modelNames) +"\\\\")
    print("Dice Score "+ "".join("& %1.3f $\pm$ %1.3f "%(m) for m in zip(mea_dic_runs, std_dic_runs) ) +"\\\\")
    print("IoU "+ "".join("& %1.3f $\pm$ %1.3f "%(m) for m in zip(mea_iou_runs, std_iou_runs) ) +"\\\\")

    
    
from .evalMetrics import evalAndComputeTVSP

def generateTVspread(models, modelNames):
    m = models
    mn = modelNames
    
    m_sdv = []
    m_t = []
    m_s = []
    for i, m in enumerate(models):
        # Evaluating all the results 
        m0_sdv , m0_t, m0_s = evalAndComputeTVSP(m)
        m_sdv.append(m0_sdv)
        m_t.append(m0_t)
        m_s.append(m0_s)
    return(np.asarray(m_sdv), np.asarray(m_t), np.asarray(m_s) )



def tableAppendixWeightDecay(SUM_2d, SUMparams, test_data, test_data_blur, test_data_noise):
    
    r = 0
    cnn0p5 = SUM_2d(*SUMparams).float().cuda()
    cnn0p5.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(r,0.5)))
    cnn0p5.eval();

    cnn0p25 = SUM_2d(*SUMparams).float().cuda()
    cnn0p25.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(r,0.25)))
    cnn0p25.eval();

    cnn0p125 = SUM_2d(*SUMparams).float().cuda()
    cnn0p125.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(r,0.125)))
    cnn0p125.eval();
    
    cnn0p05 = SUM_2d(*SUMparams).float().cuda()
    cnn0p05.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(r,0.075)))
    cnn0p05.eval();
    
    cnn0 = SUM_2d(*SUMparams).float().cuda()
    cnn0.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(r,0.125)))
    cnn0.eval();
    
    
    models_appendix = [cnn0p5, cnn0p25, cnn0p125, cnn0p05, cnn0]
    wdValues  = [0.5, 0.25, 0.125, 0.075, 0]
    
    
    dic, iou = generateTableResults(models_appendix, wdValues, test_data, display=True)
    print()
    dicb, ioub = generateTableResults(models_appendix, wdValues, test_data_blur, display=True)
    print()
    dicn, ioun = generateTableResults(models_appendix, wdValues, test_data_noise, display=True)
    
    #plt.show()
    
def tableDiceIoU_weightDecay(SUM_2d, argsSUM, test_data, no_runs):
    
    modelNames = [0.5, 0.25, 0.125, 0.075, 0]


    dic_runs = np.zeros((no_runs,len(modelNames)))
    iou_runs = np.zeros((no_runs,len(modelNames)))
    
    tds_runs = np.zeros((no_runs,len(modelNames)))
    tv_runs = np.zeros((no_runs,len(modelNames)))
    sp_runs = np.zeros((no_runs,len(modelNames)))
    
    # Iterating through all the runs
    for r in np.arange(no_runs):

        # Loading weights for each run
        cnn0p5 = SUM_2d(*argsSUM).float().cuda()
        cnn0p5.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(r,0.5)))
        cnn0p5.eval();

        cnn0p25 = SUM_2d(*argsSUM).float().cuda()
        cnn0p25.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(r,0.25)))
        cnn0p25.eval();

        cnn0p125 = SUM_2d(*argsSUM).float().cuda()
        cnn0p125.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(r,0.125)))
        cnn0p125.eval();

        cnn0p075 = SUM_2d(*argsSUM).float().cuda()
        cnn0p075.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(r,0.075)))
        cnn0p075.eval();

        cnn0 = SUM_2d(*argsSUM).float().cuda()
        cnn0.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(r,0)))
        cnn0.eval();
        
        models = [cnn0p5, cnn0p25, cnn0p125, cnn0p075, cnn0]

        # Computing Dice score and IoU
        dic, iou = generateTableResults(models, modelNames, test_data, display=False)
        dic_runs[r,:] = dic
        iou_runs[r,:] = iou
        
        tds, tv, sp = generateTVspread(models, modelNames)
        tv_runs[r,:] = tv
        sp_runs[r,:] = sp
        tds_runs[r,:] = tds

    # Computing mean and standard deviation for each of the runs
    mea_dic_runs = np.mean(dic_runs, axis=0).tolist()
    std_dic_runs = np.std(dic_runs, axis=0).tolist()
    mea_iou_runs = np.mean(iou_runs, axis=0).tolist()
    std_iou_runs = np.std(iou_runs, axis=0).tolist()
    
    mea_tv_runs = np.mean(tv_runs, axis=0).tolist()
    std_tv_runs = np.std(tv_runs, axis=0).tolist()
    mea_sp_runs = np.mean(sp_runs, axis=0).tolist()
    std_sp_runs = np.std(sp_runs, axis=0).tolist()
    mea_tds_runs = np.mean(tds_runs, axis=0).tolist()
    std_tds_runs = np.std(tds_runs, axis=0).tolist()

    # Displaying table
    print("W.D. val. "+ "".join("& %s "%(m) for m in modelNames) +"\\\\")
    print("Dice Sc. "+ "".join("& %1.3f $\pm$ %1.3f "%(m) for m in zip(mea_dic_runs, std_dic_runs) ) +"\\\\")
    print("IoU "+ "".join("& %1.3f $\pm$ %1.3f "%(m) for m in zip(mea_iou_runs, std_iou_runs) ) +"\\\\")
    
    
    print("SDS "+ "".join("& %1.1f $\pm$ %1.2f "%(m) for m in zip(mea_tds_runs, std_tds_runs) ) +"\\\\")
    print("FDV "+ "".join("& %1.1f $\pm$ %1.2f "%(m) for m in zip(mea_tv_runs, std_tv_runs) ) +"\\\\")
    print("FDS "+ "".join("& %1.1f $\pm$ %1.2f "%(m) for m in zip(mea_sp_runs, std_sp_runs) ) +"\\\\")
    
    
############################
# Table data augmentation
############################
#
def tableDiceIoU_Augmentations(SUM_2d, argsSUM, test_data, no_runs):
    
    modelNames = ["No augmentation", "Rotations", "Noise",
                  "Blur","Rot. noise and blur" ]


    dic_runs = np.zeros((no_runs,len(modelNames)))
    iou_runs = np.zeros((no_runs,len(modelNames)))
    
    tds_runs = np.zeros((no_runs,len(modelNames)))
    tv_runs = np.zeros((no_runs,len(modelNames)))
    sp_runs = np.zeros((no_runs,len(modelNames)))
    
    # Iterating through all the runs
    for r in np.arange(no_runs):

        # Loading weights for each run
        SUMsegNA = SUM_2d(*argsSUM).float().cuda()
        SUMsegNA.load_state_dict(torch.load( "./CNN_weights/SUM_SEG_NA_run%s.pyt"%r))
        #SUMsegNA.load_state_dict(torch.load( "./CNN_weights/SUM_REC_Noise_run%s.pyt"%r))
        SUMsegNA.eval();
        
        SUMseg = SUM_2d(*argsSUM).float().cuda()
        SUMseg.load_state_dict(torch.load( "./CNN_weights/SUM_SEG_run%s.pyt"%r))
        SUMseg.eval();
        
        SUMsegN = SUM_2d(*argsSUM).float().cuda()
        SUMsegN.load_state_dict(torch.load( "./CNN_weights/SUM_SEG_N_run%s.pyt"%r))
        SUMsegN.eval();
        
        SUMsegB = SUM_2d(*argsSUM).float().cuda()
        SUMsegB.load_state_dict(torch.load( "./CNN_weights/SUM_SEG_B_run%s.pyt"%r))
        SUMsegB.eval();
        
        
        SUMsegRNB = SUM_2d(*argsSUM).float().cuda()
        SUMsegRNB.load_state_dict(torch.load( "./CNN_weights/SUM_SEG_NB_run%s.pyt"%r))
        SUMsegRNB.eval();

        models = [SUMsegNA, SUMseg, SUMsegN, SUMsegB, SUMsegRNB]

        # Computing Dice score and IoU
        dic, iou = generateTableResults(models, modelNames, test_data, display=False)
        dic_runs[r,:] = dic
        iou_runs[r,:] = iou
        
        tds, tv, sp = generateTVspread(models, modelNames)
        tv_runs[r,:] = tv
        sp_runs[r,:] = sp
        tds_runs[r,:] = tds

    # Computing mean and standard deviation for each of the runs
    mea_dic_runs = np.mean(dic_runs, axis=0).tolist()
    std_dic_runs = np.std(dic_runs, axis=0).tolist()
    mea_iou_runs = np.mean(iou_runs, axis=0).tolist()
    std_iou_runs = np.std(iou_runs, axis=0).tolist()
    
    mea_tv_runs = np.mean(tv_runs, axis=0).tolist()
    std_tv_runs = np.std(tv_runs, axis=0).tolist()
    mea_sp_runs = np.mean(sp_runs, axis=0).tolist()
    std_sp_runs = np.std(sp_runs, axis=0).tolist()
    
    mea_tds_runs = np.mean(tds_runs, axis=0).tolist()
    std_tds_runs = np.std(tds_runs, axis=0).tolist()

    # Displaying table
    print("Augment. "+ "".join("& %s "%(m) for m in modelNames) +"\\\\")
    print("Dice Sc. "+ "".join("& %1.3f $\pm$ %1.3f "%(m) for m in zip(mea_dic_runs, std_dic_runs) ) +"\\\\")
    print("IoU "+ "".join("& %1.3f $\pm$ %1.3f "%(m) for m in zip(mea_iou_runs, std_iou_runs) ) +"\\\\")
    
    
    print("FDS "+ "".join("& %1.1f $\pm$ %1.2f "%(m) for m in zip(mea_tds_runs, std_tds_runs) ) +"\\\\")
    print("FDV "+ "".join("& %1.1f $\pm$ %1.2f "%(m) for m in zip(mea_tv_runs, std_tv_runs) ) +"\\\\")
    print("FDS "+ "".join("& %1.1f $\pm$ %1.2f "%(m) for m in zip(mea_sp_runs, std_sp_runs) ) +"\\\\")
