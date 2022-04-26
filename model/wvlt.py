import torch
import numpy as np
import pywt

def filter_bank_1D(wname):
	""" Returns 1D wavelet filterbank.
	wname: wavelet name (from pyWavelets)
	"""
	# returns analysis and synthesis filters concat-ed
	fb = torch.tensor(pywt.Wavelet(wname).filter_bank).float()
	wa, ws = fb[:2,:], fb[2:,:]
	return wa, ws

def filter_bank_2D(wname):
	""" Returns 2D wavelet filterbank.
	Formed as outerproduct using return from wvlt1Dfb
	wname: wavelet name (from pywt)
	Wa: analysis fb, 1 to n channels
	Ws: synthesis fb. n to 1 channels
	"""
	wa, ws = filter_bank_1D(wname)
	Wa, Ws = nonsep(wa), nonsep(ws)
	return Wa.transpose(0,1), Ws.transpose(0,1).flip(2,3)

def outerprod(u,v):
	""" Outer-product between vectors u, v
	"""
	W = torch.einsum('...i,...j->...ij',u,v)
	return W

def nonsep(w):
	""" to non-seperable fb
	Turns 1D filter bank into 2D non-seperable filter bank.
	W: n to 1 channels 2D filter bank
	"""
	w1 = torch.cat([w[:1], w[:1], w[1:], w[1:]])
	w2 = torch.cat([w, w])
	# add dim for torch kernel, flip so corr -> conv
	W  = outerprod(w1,w2)[None,:].flip(2,3)
	return W


