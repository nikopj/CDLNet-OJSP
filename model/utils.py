import torch
import torch.nn.functional as F
import numpy as np

def pre_process(x, stride):
	""" image preprocessing: stride-padding and mean subtraction.
	"""
	params = []
	# mean-subtract
	xmean = x.mean(dim=(2,3), keepdim=True)
	x = x - xmean
	params.append(xmean)
	# pad signal for stride
	pad = calc_pad_2D(*x.shape[2:], stride)
	x = F.pad(x, pad, mode='reflect')
	params.append(pad)
	return x, params

def post_process(x, params):
	""" undoes image pre-processing given params
	"""
	# unpad
	pad = params.pop()
	x = unpad(x, pad)
	# add mean
	xmean = params.pop()
	x = x + xmean
	return x

def calc_pad_1D(L, M):
	""" Return pad sizes for length L 1D signal to be divided by M
	"""
	if L%M == 0:
		Lpad = [0,0]
	else:
		Lprime = np.ceil(L/M) * M
		Ldiff  = Lprime - L
		Lpad   = [int(np.floor(Ldiff/2)), int(np.ceil(Ldiff/2))]
	return Lpad

def calc_pad_2D(H, W, M):
	""" Return pad sizes for image (H,W) to be divided by size M
	(H,W): input height, width
	output: (padding_left, padding_right, padding_top, padding_bottom)
	"""
	return (*calc_pad_1D(W,M), *calc_pad_1D(H,M))

def conv_pad(x, ks, mode):
	""" Pad a signal for same-sized convolution
	"""
	pad = (int(np.floor((ks-1)/2)), int(np.ceil((ks-1)/2)))
	return F.pad(x, (*pad, *pad), mode=mode)

def unpad(I, pad):
	""" Remove padding from 2D signalstack"""
	if pad[3] == 0 and pad[1] > 0:
		return I[..., pad[2]:, pad[0]:-pad[1]]
	elif pad[3] > 0 and pad[1] == 0:
		return I[..., pad[2]:-pad[3], pad[0]:]
	elif pad[3] == 0 and pad[1] == 0:
		return I[..., pad[2]:, pad[0]:]
	else:
		return I[..., pad[2]:-pad[3], pad[0]:-pad[1]]

