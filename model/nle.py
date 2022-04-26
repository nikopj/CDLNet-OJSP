import torch
import torch.nn.functional as F
import numpy as np
import scipy 
from scipy.stats import gamma
import model.wvlt as wvlt
import model.utils as utils

def noise_level(y, method="MAD", **kwargs):
    if method in [True, "MAD", "wvlt"]:
        return nle_mad(y)
    elif method == "PCA":
        return nle_pca(y)[0]
    else:
        raise NotImplementedError

def nle_mad(y):
    """ Median Absolute Deviation (MAD).
    Robust median estimator for noise standard deviation from 
    image y contaminated by AWGN.
    """
    hh = wvlt.filter_bank_2D('bior4.4')[0][3:4].to(y.device)
    C  = y.shape[1]
    hh = torch.cat([hh]*C) 
    HHy = F.conv2d(y, hh, stride=2, groups=C)
    sigma_hat = torch.median(HHy.abs().reshape(y.shape[0],-1), dim=1)[0] / 0.6745
    return sigma_hat.reshape(-1,1,1,1)

def nle_pca(img, patchsize=7, conf=(1-1e-6), itr=3):
    """ Translated from MATLAB code:
    Paper: Noise Level Estimation Using Weak Textured Patches Of
    A Single Noisy Image 
    http://www.ok.sc.e.titech.ac.jp/res/NLE/AWGNestimation.html
    """
    kh = torch.tensor([1/2,0,-1/2]).float().reshape(1,1,1,3)
    imgh = F.conv2d(img, kh)**2

    kv = kh.transpose(2,3)
    imgv = F.conv2d(img, kv)**2

    Dh = convmtx2(kh, patchsize, patchsize)
    Dv = convmtx2(kv, patchsize, patchsize)
    DD = Dh.T@Dh + Dv.T@Dv
    r = torch.linalg.matrix_rank(DD, hermitian=True)

    Dtr = torch.trace(DD)
    shape = r/2.0
    scale = 2.0 * Dtr / float(r)
    tau0 = gamma.ppf(conf, shape, scale=scale)

    nlevel = np.empty((img.shape[1],))
    th     = np.empty((img.shape[1],))
    num    = np.empty((img.shape[1],))

    for cha in range(1,img.shape[1]+1):
        X  = im2col(img[:,cha-1:cha], patchsize, patchsize)
        Xh = im2col(imgh[:,cha-1:cha], patchsize, patchsize-2)
        Xv = im2col(imgv[:,cha-1:cha], patchsize-2, patchsize)
        Xtr = Xh.sum(dim=0) + Xv.sum(dim=0)

        ###### nosie level estimation #####
        tau = np.inf
        if X.shape[1] < X.shape[0]:
            sig2 = 0
        else:
            cov = (X@(X.T))/(X.shape[1]-1)
            d = torch.linalg.eigvalsh(cov)
            sig2 = d[0]

        for i in range(2,itr):
        ##### weak texture selection #####
            tau = sig2 * tau0
            p = Xtr < tau
            Xtr = Xtr[p]
            X = X[:,p]

            ###### nosie level estimation #####
            if X.shape[1] < X.shape[0]:
                break
            cov = (X @ (X.T))/(X.shape[1]-1)
            d = torch.linalg.eigvalsh(cov)
            sig2 = d[0]

        nlevel[cha-1] = np.sqrt(sig2)
        th[cha-1]     = tau.item()
        num[cha-1]    = X.shape[1]
    if img.shape[1] == 1:
        return nlevel[0], th[0], num[0]
    return nlevel, th, num

def im2col(X, m, n):
    """ image to column.
    """
    return X.unfold(2,m,1).unfold(3,n,1).reshape(-1,m*n).T

def convmtx2(H, m, n):
    """ 2D convolution matrix.
    """
    s = H.shape[2:]
    T = torch.zeros((m-s[0]+1)*(n-s[1]+1), m*n, device=H.device)
    k = 1
    for i in range(1,m-s[0]+2):
        for j in range(1,n-s[1]+2):
            for p in range(1,s[0]+1):
                m1 = (i-1+p-1)*n+(j-1)+1
                m2 = (i-1+p-1)*n+(j-1)+1+s[1]-1
                A  = H[0,0,p-1,:]
                T[k-1,m1-1:m2] = A
            k = k+1
    return T
