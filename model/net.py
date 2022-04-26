import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.solvers import power_method, uball_project
from model.utils   import pre_process, post_process
from model.gabor   import ConvAdjoint2dGabor

def ST(x,t):
    """ shrinkage-thresholding operation. 
    """
    return x.sign()*F.relu(x.abs()-t)

class CDLNet(nn.Module):
    """ Convolutional Dictionary Learning Network:
    Interpretable denoising DNN with adaptive thresholds for robustness.
    """
    def __init__(self,
                 K = 3,            # num. unrollings
                 M = 64,           # num. filters in each filter bank operation
                 P = 7,            # square filter side length
                 s = 1,            # stride of convolutions
                 C = 1,            # num. input channels
                 t0 = 1e-2,        # initial threshold
                 adaptive = False, # noise-adaptive thresholds
                 init = True):     # False -> use power-method for weight init
        super(CDLNet, self).__init__()
        
        # -- OPERATOR INIT --
        self.A = nn.ModuleList([nn.Conv2d(C, M, P, stride=s, padding=(P-1)//2, bias=False)  for _ in range(K)])
        self.B = nn.ModuleList([nn.ConvTranspose2d(M, C, P, stride=s, padding=(P-1)//2, output_padding=s-1, bias=False) for _ in range(K)])
        self.D = self.B[0]                              # alias D to B[0], otherwise unused as z0 is zero
        self.t = nn.Parameter(t0*torch.ones(K,2,M,1,1)) # learned thresholds

        # set weights 
        W = torch.randn(M, C, P, P)
        for k in range(K):
            self.A[k].weight.data = W.clone()
            self.B[k].weight.data = W.clone()

        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0](x))
                L = power_method(DDt, torch.rand(1,C,128,128), num_iter=500, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if L < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()

            # spectral normalization (note: D is alised to B[0])
            for k in range(K):
                self.A[k].weight.data = self.A[k].weight.data / np.sqrt(L)
                self.B[k].weight.data = self.B[k].weight.data / np.sqrt(L)

        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        """ \ell_2 ball projection for filters, R_+ projection for thresholds
        """
        self.t.clamp_(0.0) 
        for k in range(self.K):
            self.A[k].weight.data = uball_project(self.A[k].weight.data)
            self.B[k].weight.data = uball_project(self.B[k].weight.data)

    def forward(self, y, sigma=None):
        """ LISTA + D w/ noise-adaptive thresholds
        """ 
        yp, params = pre_process(y, self.s)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma/255.0

        # LISTA
        z = ST(self.A[0](yp), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = ST(z - self.A[k](self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None):
        """ same as forward but yeilds intermediate sparse codes
        """
        yp, params = pre_process(y, self.s)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = ST(self.A[0](yp), self.t[0,:1] + c*self.t[0,1:2]); yield z
        for k in range(1, self.K):
            z = ST(z - self.A[k](self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2]); yield z
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        yield xhat

class GDLNet(nn.Module):
    """ Gabor Dictionary Learning Network:
    """
    def __init__(self,
                 K = 3,            # num. unrollings
                 M = 64,           # num. filters in each filter bank operation
                 P = 7,            # square filter side length
                 s = 1,            # stride of convolutions
                 C = 1,            # num. input channels
                 t0 = 1e-2,        # initial threshold
                 order = 1,        # mixture of gabor order
                 adaptive = False, # noise-adaptive thresholds
                 shared = "",      # which gabor parameters to share (e.g. "a_psi_w0_alpha")
                 init = True):     # False -> use power-method for weight init
        super(GDLNet, self).__init__()
        
        # -- OPERATOR INIT --
        self.A = nn.ModuleList([ConvAdjoint2dGabor(M, C, P, stride=s, order=order) for _ in range(K)])
        self.B = nn.ModuleList([ConvAdjoint2dGabor(M, C, P, stride=s, order=order) for _ in range(K)])
        self.D = self.B[0]                              # alias D to B[0], otherwise unused as z0 is zero
        self.t = nn.Parameter(t0*torch.ones(K,2,M,1,1)) # learned thresholds

        # set weights 
        alpha = torch.randn(order, M, C, 1, 1)
        a     = torch.randn(order, M, C, 2)
        w0    = torch.randn(order, M, C, 2)
        psi   = torch.randn(order, M, C)

        for AB in [self.A, self.B]:
            for k in range(K):
                AB[k].alpha.data = alpha.clone()
                AB[k].a.data     = a.clone()
                AB[k].w0.data    = w0.clone()
                AB[k].psi.data   = psi.clone()

        # Don't bother running code if initializing trained model from state-dict
        if init:
            print("Running power-method on initial dictionary...")
            with torch.no_grad():
                DDt = lambda x: self.D(self.A[0].T(x))
                L = power_method(DDt, torch.rand(1,C,128,128), num_iter=500, verbose=False)[0]
                print(f"Done. L={L:.3e}.")

                if L < 0:
                    print("STOP: something is very very wrong...")
                    sys.exit()

            # spectral normalization (note: D is alised to B[0])
            for k in range(K):
                self.A[k].alpha.data = self.A[k].alpha.data / np.sqrt(L)
                self.B[k].alpha.data = self.B[k].alpha.data / np.sqrt(L)

        # set parameters
        self.K = K
        self.M = M
        self.P = P
        self.s = s
        self.t0 = t0
        self.order = order
        self.adaptive = adaptive

    @torch.no_grad()
    def project(self):
        """ \ell_2 ball projection for filters, R_+ projection for thresholds
        """
        self.t.clamp_(0.0) 

    def forward(self, y, sigma=None):
        """ LISTA + D w/ noise-adaptive thresholds
        """ 
        yp, params = pre_process(y, self.s)

        # THRESHOLD SCALE-FACTOR c
        c = 0 if sigma is None or not self.adaptive else sigma/255.0

        # LISTA
        z = ST(self.A[0].T(yp), self.t[0,:1] + c*self.t[0,1:2])
        for k in range(1, self.K):
            z = ST(z - self.A[k].T(self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2])

        # DICTIONARY SYNTHESIS
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        return xhat, z

    def forward_generator(self, y, sigma=None):
        """ same as forward but yeilds intermediate sparse codes
        """
        yp, params = pre_process(y, self.s)
        c = 0 if sigma is None or not self.adaptive else sigma/255.0
        z = ST(self.A.T[0](yp), self.t[0,:1] + c*self.t[0,1:2]); yield z
        for k in range(1, self.K):
            z = ST(z - self.A[k].T(self.B[k](z) - yp), self.t[k,:1] + c*self.t[k,1:2]); yield z
        xphat = self.D(z)
        xhat  = post_process(xphat, params)
        yield xhat

