import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

def gabor_kernel(a, w0, psi, ks):
    """
    generate a batch of gabor filterbank via inverse width (a) and frequency (w0) params
    a   (precision):   (batch, out_chan, in_chan, 2) 
    w0  (center freq): (batch, out_chan, in_chan, 2)
    psi (phase):       (batch, out_chan, in_chan)
    h   (output):      (batch, out_chan, in_chan, ks, ks)
    """
    a   =  a[:,:,:,None,None,:]
    w0  = w0[:,:,:,None,None,:]
    psi = psi[:,:,:,None,None]

    # x spatial grid
    i = torch.arange(ks).to(a.device)
    x = torch.stack(torch.meshgrid(i,i, indexing='ij'), dim=2)[None,None,...]

    # x0 spatial center
    x0 = torch.tensor([(ks-1)/2,(ks-1)/2])[None,None,None,None,None,:]

    h = torch.exp( -torch.sum((a*(x-x0))**2, dim=-1) ) * \
        torch.cos(torch.sum(w0*(x-x0), dim=-1) + psi)
    return h

class ConvAdjoint2dGabor(nn.Module):
    """ Convolution with a Gabor kernel
    """
    def __init__(self, nic, noc, ks, stride=2, order=1):
        super(ConvAdjoint2dGabor, self).__init__()
        self.alpha = nn.Parameter(torch.randn((order, nic, noc, 1, 1))) 
        self.a     = nn.Parameter(torch.randn((order, nic, noc, 2)))
        self.w0    = nn.Parameter(torch.randn((order, nic, noc, 2)))
        self.psi   = nn.Parameter(torch.randn((order, nic, noc)))
        self.order  = order
        self.stride = stride
        self.ks = ks
        p = (ks-1)//2
        self._pad = (p,p,p,p)
        self._output_padding = nn.ConvTranspose2d(1,1,ks,stride=self.stride)._output_padding
        
    def weight(self, transpose=False):
        if transpose:
            w0, psi = -self.w0, -self.psi
        else:
            w0, psi = self.w0, self.psi
        return (self.alpha*gabor_kernel(self.a, w0, psi, self.ks)).sum(dim=0)

    def T(self, x):
        pad_x = F.pad(x, self._pad, mode='constant')
        return F.conv2d(pad_x, self.weight(transpose=True), stride=self.stride)

    def forward(self, x):
        output_size = (x.shape[0], x.shape[1], self.stride*x.shape[2], self.stride*x.shape[3])
        op = self._output_padding(x, output_size,
                                  (self.stride, self.stride),
                                  (self._pad[0], self._pad[0]),
                                  (self.ks, self.ks))

        return F.conv_transpose2d(x, self.weight(),
                                  padding = self._pad[0],
                                  stride  = self.stride,
                                  output_padding = op)
