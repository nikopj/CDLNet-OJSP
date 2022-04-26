import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms.functional import to_tensor

def img_load(path, gray=False):
	""" Load batched tensor image (1,C,H,W) from file path.
	"""
	if gray:
		return to_tensor(Image.open(path).convert('L'))[None,...]
	return to_tensor(Image.open(path))[None,...]

def awgn(input, noise_std):
	""" Additive White Gaussian Noise
	y: clean input image
	noise_std: (tuple) noise_std of batch size N is uniformly sampled 
	           between noise_std[0] and noise_std[1]. Expected to be in interval
			   [0,255]
	"""
	if not isinstance(noise_std, (list, tuple)):
		sigma = noise_std
	else: # uniform sampling of sigma
		sigma = noise_std[0] + \
		       (noise_std[1] - noise_std[0])*torch.rand(len(input),1,1,1, device=input.device)
	return input + torch.randn_like(input) * (sigma/255), sigma

def visplot(images,
	        grid_shape=None,
	        crange = (None,None),
	        primary_axis = 0,
	        titles	 = None,
	        colorbar = False,
	        cmap = 'gray'):
	""" Visual Subplot, adapted from Amir's code.
	Plots array of images in grid with shared axes.
	Very nice for zooming.
	"""
	if grid_shape is None:
		grid_shape = (1,len(images))
	fig, axs = plt.subplots(*grid_shape,sharex='all',sharey='all',squeeze=False)
	nrows, ncols = grid_shape
	# fill grid row-wise or column-wise
	if primary_axis == 1:
		indfun = lambda i,j: j*nrows + i
	else:
		indfun = lambda i,j: i*ncols + j
	im_list = []
	for ii in range(nrows):
		for jj in range(ncols):
			ind = indfun(ii,jj)
			if ind < len(images):
				if type(images[ind])==torch.Tensor:
					img = images[ind].detach().permute(1,2,0).squeeze()
				else:
					img = images[ind].squeeze()
				im = axs[ii,jj].imshow(img,
				                       cmap   = cmap,
				                       aspect = 'equal',
				                       interpolation = None,
				                       vmin = crange[0],
				                       vmax = crange[1])
				if colorbar:
					fig.colorbar(im,
					             ax       = axs[ii,jj],
					             fraction = 0.046,
					             pad      = 0.04)
			axs[ii,jj].axis('off')
			if (titles is not None) and (ind < len(titles)):
				axs[ii,jj].set_title(titles[ind])
	return fig
