import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift


from .evalMetrics import evalAndComputeTVSP, impRes


##################################
# Figure segmentation
##################################
def OneRow(cnn, dataset, i, axs, isUNet=True):
    with torch.no_grad():
        im, mk = dataset.__getitem__(i)
        
        #print(im.shape)
        if not isUNet:
            ev, delta = cnn( im.unsqueeze(0), thr=1 )[0:2]
            rec =  cnn( im.unsqueeze(0), thr= 0.0 )[1]
        else:
            ev, delta = cnn( im.unsqueeze(0) )
            rec =  ev.cpu()*torch.tensor([(np.NaN)])
            
        
            
        
    inp = im.detach().cpu()[0].numpy()
    res = ev.detach().cpu()[0,0].numpy()
    gtr = mk.detach().cpu()[0].numpy()
    rec = rec.cpu()[0,0].numpy()
    delta = delta.cpu()[0,0].numpy()
    mk = mk.detach().cpu()[0].numpy()
    
    
    #from scipy.stats import mode
    # Removing offsets
    #offset = mode(delta[np.where(inp<0.25)])[0]
    #delta = delta - offset
    
    
    #from scipy.ndimage.morphology import binary_erosion
    ## Making the mask narrower for avoiding outliers in the borders in the U-Net
    mk2 = mk #binary_erosion(mk, structure=np.ones((5,5)) ).copy()
    
    
    # Scaling the reconstruction for better comparison with input
    rec2 = rec.copy()*mk2;
    inp2 = inp.copy()*mk2
    alpha = np.sum(inp2*rec2 )/np.sum(rec2*rec2 + 1e-4)
    
    # Scaling the estimated anomaly for better comparison with the input
    lamb = 1.5
    delta2 = delta.copy()*mk2
    beta = np.sum(inp2 *delta2) / ((1+lamb)*np.sum(delta2**2) +1e-4 )
    
    
    # Plotting the results
    # Input
    vmin_gray = -0.25
    img = axs[0].imshow(inp[:,:], vmax = 1, vmin = vmin_gray,cmap="gray")
    axs[0].yaxis.set_major_locator(plt.NullLocator())
    axs[0].xaxis.set_major_formatter(plt.NullFormatter())
    axs[0].axis('off')

    # Input
    axs[1].imshow(alpha*rec[:,:],cmap="gray", vmax = 1, vmin = vmin_gray )#
    axs[1].yaxis.set_major_locator(plt.NullLocator())
    axs[1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1].axis('off')
    
    # Input #beta*
    axs[2].imshow( beta*delta, cmap="gray", vmax = 1,vmin= vmin_gray)
    axs[2].yaxis.set_major_locator(plt.NullLocator())
    axs[2].xaxis.set_major_formatter(plt.NullFormatter())
    axs[2].axis('off')

    # Result
    axs[3].imshow(inp[:,:], vmax = 1, vmin = vmin_gray,cmap="gray")
    axs[3].yaxis.set_major_locator(plt.NullLocator())
    axs[3].xaxis.set_major_formatter(plt.NullFormatter())
    axs[3].axis('off')

    seg= axs[3].imshow(res[:,:], vmax = 1, vmin = 0, cmap='jet', alpha=0.5)
    axs[3].yaxis.set_major_locator(plt.NullLocator())
    axs[3].xaxis.set_major_formatter(plt.NullFormatter())

    # Ground Truth
    axs[4].imshow(inp[:,:], vmax = 1, vmin = vmin_gray,cmap="gray")
    axs[4].yaxis.set_major_locator(plt.NullLocator())
    axs[4].xaxis.set_major_formatter(plt.NullFormatter())

    axs[4].imshow(gtr[:,:], vmax = 1, vmin = 0, cmap='jet', alpha=0.5)
    axs[4].yaxis.set_major_locator(plt.NullLocator())
    axs[4].xaxis.set_major_formatter(plt.NullFormatter())
    axs[4].axis('off')
    return(img, seg)
    
def figureSegmSlices(models, modelNames, isUNet, whichSubject, testSet, sc, fs, fname):
    nr = len(models)
    
    fig, axs = plt.subplots(nrows=nr, ncols=5,figsize=(sc*5,sc*(nr+0.1) ) )
    for i, m in enumerate(models):
        img, seg = OneRow(models[i], dataset= testSet,
               i=whichSubject, axs=axs[i], isUNet=isUNet[i])

        plt.text(0.1, 0.5, modelNames[i],
            color="yellow",
            horizontalalignment='center', verticalalignment='center',
                 rotation="vertical", fontSize=fs, transform = axs[i,0].transAxes)
    
    plt.text(0.5, 0.95, 'Input',
        color="yellow", horizontalalignment='center', verticalalignment='top', fontSize=fs,
        transform = axs[0,0].transAxes)
    plt.text(0.5, 0.95,  r'$\mathregular{Lin. ED~}(\mathregular{ED}_{\mathregular{S}(\alpha=0)}(\cdot))$',
        color="yellow",
        horizontalalignment='center', verticalalignment='top', fontSize=fs, transform = axs[0,1].transAxes)
    plt.text(0.5, 0.95, 'Inp. sig. O.L.',
        color="yellow",
        horizontalalignment='center', verticalalignment='top', fontSize=fs, transform = axs[0,2].transAxes)

    plt.text(0.5, 0.95, 'Est. segmentation',
        color="yellow",
        horizontalalignment='center', verticalalignment='top', fontSize=fs, transform = axs[0,3].transAxes)
    plt.text(0.5, 0.95, 'True segmentation',
        color="yellow",
        horizontalalignment='center', verticalalignment='top', fontSize=fs, transform = axs[0,4].transAxes)

    cbaxesTop = fig.add_axes([0.91, 0.60, 0.02, 0.24])  # 
    fig.colorbar(img, orientation='vertical', cax=cbaxesTop)
    
    cbaxesBot = fig.add_axes([0.91, 0.20, 0.02, 0.24])  # 
    fig.colorbar(seg, orientation='vertical', cax=cbaxesBot)
    
    plt.subplots_adjust(wspace=0, hspace= 0.0)
    plt.savefig(fname,bbox_inches='tight')
    plt.show()


##################################
# Figures of impulse response
##################################



def iRplotCol(cnn, axs, isRef=True, title="", vminTop=-0.2, vmaxTop=2.0, vminBot=-0.2, vmaxBot=2.0):    

    respnp, impnp = impRes(cnn, isRef)
    
    # Center pixel always positive
    sign = np.sign( np.sum(respnp*impnp) )
    respnp*=sign
    
    # Normalizing energy
    respnp/= np.sqrt(np.sum(respnp**2))
    
    # Computing energy 
    ener_resp = np.sum(respnp**2)
    ener_center_resp = np.sum( (respnp*impnp)**2)    
    
    # Making the figure
    labelsize = 25
    labelcolor="y"
    
    offs = 32
    imgt = axs[0].imshow(respnp[offs:-offs,offs:-offs],cmap="hot",vmin=vminTop,vmax=vmaxTop)
    axs[0].yaxis.set_major_locator(plt.NullLocator())
    axs[0].xaxis.set_major_locator(plt.NullLocator())

    # Frequency response
    ft = np.absolute(fftshift(fft2( respnp, (256,256) ) ) )
    imgb =axs[1].imshow(ft ,cmap="magma" ,vmin=vminBot, vmax=vmaxBot)
    axs[1].yaxis.set_major_locator(plt.NullLocator())
    axs[1].xaxis.set_major_locator(plt.NullLocator())

    fs=15
    plt.text(0.5, 0.95, title,
        color="yellow",
        horizontalalignment='center', verticalalignment='top',
             fontSize=fs, transform = axs[0].transAxes)
    return([imgt, imgb])


def impulseResponses(model, modelWeightPaths, model_names, figure_name):
    sc=3.0
    fig, axs = plt.subplots(nrows=2, ncols=3,figsize=(sc*3.0,sc*2.05))
    fs=14
    

    imgTop, imgBot = iRplotCol([],      [axs[0,0], axs[1,0]], isRef=True,  title="Input impulse",  
                    vminTop=-0.035, vmaxTop = 0.50,  vminBot=0, vmaxBot=2.5)
    tds, fdv, fds = evalAndComputeTVSP([], isRef=True)
    plt.text(0.0, 0.15, "SD spread: %1.2f"%tds, color="yellow",
            fontSize=fs, transform = axs[0,0].transAxes)
    plt.text(0.0, 0.15, "FD spread: %1.2f"%fds, color="yellow",
            fontSize=fs, transform = axs[1,0].transAxes)
    plt.text(0.0, 0.05, "FD variation: %1.2f"%fdv, color="yellow",
            fontSize=fs, transform = axs[1,0].transAxes)
        
    
    for i, mwp in enumerate(modelWeightPaths):
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load(mwp ))
        model.eval()
        
        iRplotCol(model, [axs[0,i+1], axs[1,i+1]], isRef=False, title= model_names[i], 
                  vminTop=-0.035, vmaxTop = 0.50,  vminBot=0, vmaxBot=2.5)
        tds, fdv, fds = evalAndComputeTVSP(model)
        plt.text(0.0, 0.15, "SD spread: %1.2f"%tds, color="yellow",
            fontSize=fs, transform = axs[0,i+1].transAxes)
        plt.text(0.0, 0.15, "FD spread: %1.2f"%fds, color="yellow",
                fontSize=fs, transform = axs[1,i+1].transAxes)
        plt.text(0.0, 0.05, "FD variation: %1.2f"%fdv, color="yellow",
                fontSize=fs, transform = axs[1,i+1].transAxes)

    plt.text(0.1, 0.6, 'Imp. response',
            color="yellow",
            horizontalalignment='center', verticalalignment='center', rotation="vertical",
            fontSize=fs, transform = axs[0,0].transAxes)
    plt.text(0.1, 0.6, 'Freq. response',
            color="yellow",
            horizontalalignment='center', verticalalignment='center', rotation="vertical",
            fontSize=fs, transform = axs[1,0].transAxes)

    del model
    torch.cuda.empty_cache()
    
    cbaxesTop = fig.add_axes([0.91, 0.60, 0.02, 0.24])  # 
    fig.colorbar(imgTop, orientation='vertical', cax=cbaxesTop)
    
    cbaxesBot = fig.add_axes([0.91, 0.20, 0.02, 0.24])  # 
    fig.colorbar(imgBot, orientation='vertical', cax=cbaxesBot)

    plt.subplots_adjust(wspace=0, hspace= 0.0)
    plt.savefig(figure_name,bbox_inches='tight')
    plt.show()
    
    
from .evalMetrics import fd_spread, fd_flatness


def figure1(SUM_2d, unet_2d, unetNoBN_2d, SUMparams, UNetParams, run, test_data, whichSample):
    torch.cuda.empty_cache()
    
    SUMrec = SUM_2d(*SUMparams).float().cuda()
    SUMseg = SUM_2d(*SUMparams).float().cuda()
    uNet = unet_2d(*UNetParams).float().cuda()
    uNetNoBN = unetNoBN_2d(*UNetParams).float().cuda()
    
    
    SUMseg.load_state_dict(torch.load( "./CNN_weights/SUM_SEG_run%s.pyt"%run))
    SUMseg.eval();

    SUMrec.load_state_dict(torch.load( "./CNN_weights/SUM_REC_run%s.pyt"%run))
    SUMrec.eval();

    uNet.load_state_dict(torch.load( "./CNN_weights/unet_run%s.pyt"%run))
    uNet.eval();
    
    uNetNoBN.load_state_dict(torch.load( "./CNN_weights/unet_NoBN_run%s.pyt"%run))
    uNetNoBN.eval();
    
    modelsFig = [SUMseg, SUMrec, uNet, uNetNoBN]
    
    modelNamesFig = [
    '$\mathregular{SUM_{Seg}}$',
    '$\mathregular{SUM_{Rec}}$',
    'U-Net', 'U-'+'$\mathregular{Net_{NoBN}}$']

    isUNet = [False, False, True, True] # 
    
    # Font size and scale
    fs = 14
    sc =3
    
    figureSegmSlices(modelsFig, modelNamesFig, isUNet, whichSample, test_data,  sc, fs, fname="results/normal_v2_%s.pdf"%str(whichSample) )
    
    del SUMseg
    del SUMrec
    del uNet
    del uNetNoBN
    
    return None


def irSegRec(SUM_2d, SUMparams, weightPaths, modelNames, figureName):
    SUM = SUM_2d(*SUMparams).float().cuda()
    impulseResponses(
        SUM,
        weightPaths,
        modelNames,
        figure_name=figureName)
    del SUM
    return None
    
    
    
##################################
# Figure 5
##################################
def OneRowDeno(cnn, dataset, i, axs, isDeno=True):
    with torch.no_grad():
        im, mk = dataset.__getitem__(i)
        
        if isDeno:
            ims = im.shape
            noi = 0.1*np.random.normal(np.zeros( (ims[1],ims[2]) ),
                                        np.ones( (ims[1],ims[2]) ))
            noi = torch.from_numpy(noi).unsqueeze(0).cuda().to(torch.float)
            inp = im + noi
        else:
            inp = im
        
        ev, delta = cnn( inp.unsqueeze(0), thr=1 )[0:2]
        rec =  cnn( inp.unsqueeze(0), thr= 0.0 )[1]
            
    
    inp = inp.detach().cpu()[0].numpy()
    res = ev.detach().cpu()[0,0].numpy()
    gtr = mk.detach().cpu()[0].numpy()
    rec = rec.cpu()[0,0].numpy()
    delta = delta.cpu()[0,0].numpy()
    mk = mk.detach().cpu()[0].numpy()
    
    
    mk2 = mk 

    # Scaling the reconstruction for better comparison with input
    inp2 = inp.copy()
    rec2 = rec.copy()*mk2;
    alpha = np.sum(inp2*rec2 )/np.sum(rec2*rec2 + 1e-4)

    # Scaling the estimated anomaly for better comparison with the input
    lamb = 1.5
    delta2 = delta.copy()*mk2
    beta = np.sum(inp2 *delta2) / ((1+lamb)*np.sum(delta2**2) +1e-4 )
    if isDeno:
        beta = np.sign(delta)
    
    
    # Plotting the results
    # Input
    vmin_gray = -0.25
    img = axs[0].imshow(inp[:,:], vmax = 1, vmin = vmin_gray,cmap="gray")
    axs[0].yaxis.set_major_locator(plt.NullLocator())
    axs[0].xaxis.set_major_formatter(plt.NullFormatter())
    axs[0].axis('off')

    # Input
    axs[1].imshow(alpha*rec[:,:],cmap="gray", vmax = 1, vmin = vmin_gray )#
    axs[1].yaxis.set_major_locator(plt.NullLocator())
    axs[1].xaxis.set_major_formatter(plt.NullFormatter())
    axs[1].axis('off')
    
    # Input #beta*
    axs[2].imshow( beta*delta, cmap="gray", vmax = 1,vmin= vmin_gray)
    axs[2].yaxis.set_major_locator(plt.NullLocator())
    axs[2].xaxis.set_major_formatter(plt.NullFormatter())
    axs[2].axis('off')
    
    if isDeno:
        
        # Result
        axs[3].imshow( res, vmax = 1, vmin = vmin_gray,cmap="gray")
        axs[3].yaxis.set_major_locator(plt.NullLocator())
        axs[3].xaxis.set_major_formatter(plt.NullFormatter())
        axs[3].axis('off')

        # Ground Truth
        axs[4].imshow(im.detach().cpu()[0].numpy(), vmax = 1, vmin = vmin_gray,cmap="gray")
        axs[4].yaxis.set_major_locator(plt.NullLocator())
        axs[4].xaxis.set_major_formatter(plt.NullFormatter())
        axs[4].axis('off')
        return(img)

    else:
        # Result
        axs[3].imshow(inp[:,:], vmax = 1, vmin = vmin_gray,cmap="gray")
        axs[3].yaxis.set_major_locator(plt.NullLocator())
        axs[3].xaxis.set_major_formatter(plt.NullFormatter())
        axs[3].axis('off')

        seg= axs[3].imshow(res[:,:], vmax = 1, vmin = 0, cmap='jet', alpha=0.5)
        axs[3].yaxis.set_major_locator(plt.NullLocator())
        axs[3].xaxis.set_major_formatter(plt.NullFormatter())
        axs[3].axis('off')

        # Ground Truth
        axs[4].imshow(inp[:,:], vmax = 1, vmin = vmin_gray,cmap="gray")
        axs[4].yaxis.set_major_locator(plt.NullLocator())
        axs[4].xaxis.set_major_formatter(plt.NullFormatter())
        axs[4].axis('off')

        axs[4].imshow(gtr[:,:], vmax = 1, vmin = 0, cmap='jet', alpha=0.5)
        axs[4].yaxis.set_major_locator(plt.NullLocator())
        axs[4].xaxis.set_major_formatter(plt.NullFormatter())
        axs[4].axis('off')
        return(img, seg)


def figureSegmAndDenoSlices(models, modelNames, isDeno, whichSubject, testSet, sc, fs, fname):
    nr = len(models)
    
    fig, axs = plt.subplots(nrows=nr, ncols=5,figsize=(sc*5,sc*(nr+0.1) ) )
    for i, m in enumerate(models):

        oi = OneRowDeno(models[i], dataset= testSet,
               i=whichSubject, axs=axs[i], isDeno=isDeno[i])
        
        plt.text(0.1, 0.5, modelNames[i],
            color="yellow",
            horizontalalignment='center', verticalalignment='center',
                 rotation="vertical", fontSize=fs, transform = axs[i,0].transAxes)
    
    plt.text(0.5, 0.95, 'Input',
        color="yellow", horizontalalignment='center', verticalalignment='top', fontSize=fs,
        transform = axs[0,0].transAxes)
    plt.text(0.5, 0.95,  r'$\mathregular{Lin. ED~}(\mathregular{ED}_{\mathregular{S}(\alpha=0)}(\cdot))$',
        color="yellow",
        horizontalalignment='center', verticalalignment='top', fontSize=fs, transform = axs[0,1].transAxes)
    plt.text(0.5, 0.95, 'Inp. sig. O.L.',
        color="yellow",
        horizontalalignment='center', verticalalignment='top', fontSize=fs, transform = axs[0,2].transAxes)

    plt.text(0.5, 0.95, 'Estimate',
        color="yellow",
        horizontalalignment='center', verticalalignment='top', fontSize=fs, transform = axs[0,3].transAxes)
    plt.text(0.5, 0.95, 'Ground truth',
        color="yellow",
        horizontalalignment='center', verticalalignment='top', fontSize=fs, transform = axs[0,4].transAxes)

    #cbaxesTop = fig.add_axes([0.91, 0.60, 0.02, 0.24])  # 
    #fig.colorbar(img, orientation='vertical', cax=cbaxesTop)
    
    
    plt.subplots_adjust(wspace=0, hspace= 0.0)
    plt.savefig(fname,bbox_inches='tight')
    plt.show()
    
def figure5(SUM_2d, SUMparams, SUMsegWeights, SUMdenoWeights, modelNamesFig, whichSubject, test_data, fname):

    torch.cuda.empty_cache()
    
    SUMseg = SUM_2d(*SUMparams).float().cuda()
    SUMseg.load_state_dict(torch.load( SUMsegWeights ))
    SUMseg.eval();

    SUMdeno = SUM_2d(*SUMparams).float().cuda()
    SUMdeno.load_state_dict(torch.load( SUMdenoWeights))
    SUMdeno.eval();
    
    # Scale and font size
    sc=3
    fs = 14
    
    isDeno = [False, True]
    modelsFig = [SUMseg, SUMdeno]
    figureSegmAndDenoSlices(modelsFig, modelNamesFig, isDeno, whichSubject, test_data,  sc, fs, fname=fname)
    
    
######################################################################################################
######################################################################################################
######################################################################################################
# APPENDICES
######################################################################################################
######################################################################################################
######################################################################################################


def ablationStudyWeightDecay(SUM_2d, SUMparams, dataset, whichSubject, wDVals):
    sc=3.5
    fig, axs = plt.subplots(nrows=2, ncols=len(wDVals)+1,figsize=(sc*(len(wDVals)+1),sc*2.05))
    fs=14

    ######################
    # Impulse Responses
    ######################
    imgTop, imgCen = iRplotCol([],      [axs[0,0], axs[1,0]], isRef=True,  title="Input impulse",
                               vminTop=-0.035, vmaxTop = 0.30, vminBot=0, vmaxBot=7)
    tds, fdv, fds = evalAndComputeTVSP([], isRef=True)
    plt.text(0.0, 0.15, "SD spread: %1.2f"%tds, color="yellow",
            fontSize=fs, transform = axs[0,0].transAxes)
    plt.text(0.0, 0.15, "FD spread: %1.2f"%fds, color="yellow",
            fontSize=fs, transform = axs[1,0].transAxes)
    plt.text(0.0, 0.05, "FD variation: %1.2f"%fdv, color="yellow",
            fontSize=fs, transform = axs[1,0].transAxes)
        
    plt.text(0.1, 0.6, 'Imp. response',
            color="yellow",
            horizontalalignment='center', verticalalignment='center', rotation="vertical",
            fontSize=fs, transform = axs[0,0].transAxes)
    plt.text(0.1, 0.6, 'Freq. response',
            color="yellow",
            horizontalalignment='center', verticalalignment='center', rotation="vertical",
            fontSize=fs, transform = axs[1,0].transAxes)
    
    for i, wd in enumerate(wDVals):
        
        run=0
        
        cnn = SUM_2d(*SUMparams).float().cuda()
        cnn.load_state_dict(torch.load( "./CNN_weights/SUM_SEGNA_run%s_wd%s.pyt"%(run, wd) ))
        cnn.eval();
        
        iRplotCol(cnn, [axs[0,i+1], axs[1,i+1]], isRef=False, title= "$\mathregular{SUM_{Seg}}$"+", w. decay %s"%str(wd),
                 vminTop=-0.035, vmaxTop = 0.30, vminBot=0, vmaxBot=7)
        tds, fdv, fds = evalAndComputeTVSP(cnn)
        plt.text(0.0, 0.15, "SD spread: %1.2f"%tds, color="yellow",
                fontSize=fs, transform = axs[0,i+1].transAxes)
        plt.text(0.0, 0.15, "FD spread: %1.2f"%fds, color="yellow",
                fontSize=fs, transform = axs[1,i+1].transAxes)
        plt.text(0.0, 0.05, "FD variation: %1.2f"%fdv, color="yellow",
                fontSize=fs, transform = axs[1,i+1].transAxes)
    
        del cnn
    
    #############
    # Colorbars
    ##############
    
    cbaxesTop = fig.add_axes([0.91, 0.67, 0.02, 0.15])  # 
    fig.colorbar(imgTop, orientation='vertical', cax=cbaxesTop)
    
    cbaxesCen = fig.add_axes([0.91, 0.42, 0.02, 0.15])  # 
    fig.colorbar(imgCen, orientation='vertical', cax=cbaxesCen)

    plt.subplots_adjust(wspace=0, hspace= 0.0)
    plt.savefig("results/irReconVsWD.pdf",bbox_inches='tight')
    plt.show()
    
    
def impulRespAug(SUM_2d, SUMparams, r):
    sc=3.5
    fig, axs = plt.subplots(nrows=2, ncols=6,figsize=(sc*6.0,sc*2.05))
    fs=16

    
    imgTop, imgBot = iRplotCol([], [axs[0,0], axs[1,0]], isRef=True,  title="Input impulse",  
                    vminTop=-0.025, vmaxTop = 0.30,  vminBot=0, vmaxBot=7)
    tds, fdv, fds = evalAndComputeTVSP([], isRef=True)
    plt.text(0.0, 0.15, "SD spread: %1.2f"%tds, color="yellow",
            fontSize=fs, transform = axs[0,0].transAxes)
    plt.text(0.0, 0.15, "FD spread: %1.2f"%fds, color="yellow",
            fontSize=fs, transform = axs[1,0].transAxes)
    plt.text(0.0, 0.05, "FD variation: %1.2f"%fdv, color="yellow",
            fontSize=fs, transform = axs[1,0].transAxes)
    

    model_dirs = [
        "./CNN_weights/SUM_SEG_NA_run%s.pyt"%r,
        "./CNN_weights/SUM_SEG_run%s.pyt"%r,
        "./CNN_weights/SUM_SEG_N_run%s.pyt"%r,
        "./CNN_weights/SUM_SEG_B_run%s.pyt"%r,
        "./CNN_weights/SUM_SEG_NB_run%s.pyt"%r,
    ]
    
    model_names = [
        "No augmentations",
        "Rotations + mirroring",
        "Noise",
        "Blur",
        "Rot. + blur + noise",
    ]
    
    for i, m in enumerate(model_dirs):
        torch.cuda.empty_cache()
        SUM = SUM_2d(*SUMparams).cuda()
        SUM.eval();
        SUM.load_state_dict( torch.load( m ) )
        iRplotCol(SUM, [axs[0,i+1], axs[1,i+1]], isRef=False, title= model_names[i],  
                  vminTop=-0.035, vmaxTop = 0.30,  vminBot=0, vmaxBot=7)
        tds, fdv, fds = evalAndComputeTVSP(SUM)
        plt.text(0.0, 0.15, "SD spread: %1.2f"%tds, color="yellow",
                fontSize=fs, transform = axs[0,i+1].transAxes)
        plt.text(0.0, 0.15, "FD spread: %1.2f"%fds, color="yellow",
                fontSize=fs, transform = axs[1,i+1].transAxes)
        plt.text(0.0, 0.05, "FD variation: %1.2f"%fdv, color="yellow",
                fontSize=fs, transform = axs[1,i+1].transAxes)
        torch.cuda.empty_cache()

        del SUM

    
    
    plt.text(0.1, 0.6, 'Imp. response',
            color="yellow",
            horizontalalignment='center', verticalalignment='center', rotation="vertical",
            fontSize=fs, transform = axs[0,0].transAxes)
    plt.text(0.1, 0.6, 'Freq. response',
            color="yellow",
            horizontalalignment='center', verticalalignment='center', rotation="vertical",
            fontSize=fs, transform = axs[1,0].transAxes)
    
    cbaxesTop = fig.add_axes([0.91, 0.60, 0.02, 0.24])  # 
    fig.colorbar(imgTop, orientation='vertical', cax=cbaxesTop)
    
    cbaxesBot = fig.add_axes([0.91, 0.20, 0.02, 0.24])  # 
    fig.colorbar(imgBot, orientation='vertical', cax=cbaxesBot)

    plt.subplots_adjust(wspace=0, hspace= 0.0)
    plt.savefig("results/irAugm.pdf",bbox_inches='tight')
    plt.show()
