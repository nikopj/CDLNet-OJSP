from os import path, listdir
from glob import glob
from PIL import Image
import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from tqdm import tqdm

class MyDataset(data.Dataset):
	def __init__(self, root_dirs, transform, load_color=False):
		self.image_paths = []
		self.image_list = []

		for cur_path in root_dirs:
			self.image_paths += [path.join(cur_path, file) \
				for file in listdir(cur_path) \
				if file.endswith(('tif','tiff','png','jpg','jpeg','bmp'))]

		print(f"Loading {root_dirs}:")
		for i in tqdm(range(len(self.image_paths))):
			if load_color:
				self.image_list.append(Image.open(self.image_paths[i]))
			else:
				self.image_list.append(Image.open(self.image_paths[i]).convert('L'))

		self.root_dirs = root_dirs
		self.transform = transform

	def __len__(self):
		return len(self.image_paths)

	def __getitem__(self, idx):
		return self.transform(self.image_list[idx])

def getDataLoader(dir_list, batch_size=1, load_color=False, crop_size=None, test=True):
	if test:
		xfm = transforms.ToTensor()
	else:
		xfm = transforms.Compose([transforms.RandomCrop(crop_size),
		                          transforms.RandomHorizontalFlip(),
		                          transforms.RandomVerticalFlip(),
		                          transforms.ToTensor()])

	return data.DataLoader(MyDataset(dir_list, xfm, load_color),
	                       batch_size = batch_size,
	                       drop_last  = (not test),
	                       shuffle    = (not test))

def getFitLoaders(trn_path_list =['CBSD432'],
	              val_path_list=['Kodak'],
	              tst_path_list=['CBSD68'],
	              crop_size  = 128,
	              batch_size = [10,1,1],
	              load_color = False):

	if type(batch_size) is int:
		batch_size = [batch_size, 1, 1]

	dataloaders = {'train': getDataLoader(trn_path_list, 
                                          batch_size[0], 
                                          load_color, 
                                          crop_size=crop_size, 
                                          test=False),
	               'val':   getDataLoader(val_path_list, 
                                          batch_size[1], 
                                          load_color, 
                                          test=True),
	               'test':  getDataLoader(tst_path_list, 
                                          batch_size[2], 
                                          load_color, 
                                          test=True)}
	return dataloaders

