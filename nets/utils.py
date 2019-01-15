from __future__ import print_function
import os, csv, sys, gzip, torch, time, pickle, argparse
import torch.nn as nn
import numpy as np
import scipy.misc
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pdb
from itertools import islice
import pandas as pd
from numpy import genfromtxt
from collections import defaultdict
import scipy.io as sio

def load_mnist(dataset, dataroot_dir = './data'):
	data_dir = os.path.join(dataroot_dir, dataset)

	def extract_data(filename, num_data, head_size, data_size):
		with gzip.open(filename) as bytestream:
			bytstream.read(head_size)
			buf = bytestream.read(data_size * num_data)
			data = np.frombuffer(buf, dtype=np.unit8).astype(np.float)
		return data
	
	data = extract_data(data_dir + '/train-images-idx3-ubyte.gz', 60000, 16, 28*28)
	trX = data.reshape((60000, 28, 28, 1))

	data = extract_data(data_dir + '/train-labels-idx1-ubyte.gz', 60000, 8, 1)
	trY = data.reshape((60000))

	data = extract_data(data_dir + '/t10k-images-idx3-ubyte.gz', 10000, 16, 28*28)
	teX = data.reshape((10000, 28, 28, 1))

	data = extract_data(data_dir+'/t10k-labels-idx1-ubyte.gz', 10000, 8, 1)
	teY = data.reshape((10000))

	x = np.concatenate((trX, teX), axis=0)
	y = np.concatenate((trY, teY), axis=0).astype(np.int)
	
	seed = 547
	np.random.seed(seed)
	np.random.shuffle(x)
	np.random.seed(seed)
	np.random.shuffle(y)

	y_vec = np.zeros((len(y), 10),dtype=np.float)
	for i , label in enumerate(y):
		y_vec[i, y[i]] = 1
	
	x = x.transpose(0, 3, 1, 2) / 255.

	x = torch.from_numpy(x).type(torch.FloatTensor)
	#x = torch.FloatTensor(x)
	y_vec = torch.from_numpy(y_vec).type(torch.FloatTensor)
	#y = torch.FloatTensor(y_vec)
	return x , y_vec


def CustomDataLoader(path, transform, batch_size, shuffle):
	dset = datasets.ImageFolder(path, transform)
	data_loader = torch.utils.data.DataLoader(dset, batch_size, shuffle)

	return data_loader


class CelebA(Dataset):
	def __init__(self, root_dir, transform = None, _type = None):
		self.filenames = []
		self.root_dir = root_dir
		self.transform = transform
		self.type = _type
		cnt = 15000


		print('Loadiong CelebA metatdata...')
		sys.stdout.flush()
		time_start = time.time()
		
		fname_cache = 'CelebA_cache.txt'
		if os.path.exists(fname_cache):
			self.filenames = open(fname_cache).read().splitlines()
			print('Already cache file exists! Load from here ....')
		else:
			path = os.path.join(root_dir)

			self.filenames = [os.path.join(dirpath,f) for(dirpath, dirnames, files)in os.walk(path) for f in files if f.endswith('.jpg')][:5000]

			#self.filenames = self.filenames[:cnt]
			print('=====Making cache_file.txt=======')
			with open(fname_cache, 'w') as f:
				for fname in self.filenames:
					f.write(fname+'\n')
			print('Done! cached in {}'.format(fname_cache))

	def __len__(self):
		return len(self.filenames)


	def __getitem__(self, idx):
		filename = self.filenames[idx]
		image = Image.open(filename)
		image = image.convert('RGB')
		image = self.transform(image)
		return image


class CUB(Dataset):
	def __init__(self, root_dir, transform=None, _type=None):
		self.filenames = []
		self.root_dir = root_dir
		self.transform = transform
		self.type = _type

		print('Loading CUB metadata...')
		sys.stdout.flush()
		time_start = time.time()

		fname_cache = 'CUB_cache.txt'
		if os.path.exists(fname_cache):
			self.filenames = open(fname_cache).read().splitlines()
			print('Already cache file exists! Load from here ....')
		else:
			path = os.path.join(self.root_dir)
			self.filenames = [os.path.join(dirpath, f) for (dirpath, dirnames, files) in os.walk(path) for f in files if f.endswith('.jpg')]

			print('======Making cache_file.txt=======')
			with open(fname_cache, 'w') as f:
				for fname in self.filenames:
					f.write(fname+'\n')
			print('Done! cached in {}'.format(fname_cache))
		
	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		filenames = self.filenames[idx]
		image = Image.open(filenames).convert('RGB')
		if self.transform:
			image = self.transform(image)
		return image

class ImageNet(Dataset):
	def __init__(self, root_dir, transform = None, _type = None, num_cls = None):
		self.filenames = []
		self.root_dir = root_dir#../../../ImageNet/ILSVRC/Data/Det
		self.transform = transform
		self.type = _type
		self.num_cls = num_cls+1

		print('Loading ImageNet metadata...')
		sys.stdout.flush()
		time_start = time.time()

		#make cache text file
		fname_cache = 'ImageNet_cache.txt'
		if os.path.exists(fname_cache):
			self.filenames = open(fname_cache).read().splitlines()
			print('Already cache file exists! Load from here...')
		else:
			if self.type == 'train':
				path = os.path.join(root_dir, self.type,'ILSVRC2013_train')
			else:
				path = os.path.join(root_dir, self.type)
			
			self.filenames = [ os.path.join(dirpath,f) for _,( dirpath, dirnames,files) in zip(range(self.num_cls), os.walk(path)) for f in files if f.endswith('.JPEG')]
			print('---Making cache_file.txt----')

			with open(fname_cache, 'w') as f:
				for fname in self.filenames:
					f.write(fname+'\n')
			print('Done! cached in {}'.format(fname_cache))



		#get ImageNet file path
		self.cls = sorted(set( [os.path.basename(f).split('_')[0] for f in self.filenames]))
		self.cls_map = {}
		for i, cls in enumerate(self.cls):
			self.cls_map[cls] = i	
		print('Loading ImageNet done!')

		
	
	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		#load image
		filename = self.filenames[idx]
		image = Image.open(filename)
		image = image.convert('RGB')
		cls_ = os.path.basename(filename).split('_')[0]
		if self.transform:
			image = self.transform(image)

		cls = self.cls_map[cls_]
		
		return image, cls


def print_network(net):
	num_params = 0
	for param in net.parameters():
		num_params += param.numel()
	print(net)
	print('Total number of parameters: %d' % num_params)

def save_images(images, size, image_path):
	return imsave(images, size, image_path)

def imsave(images, size, path):
	image = np.squeeze(merge(images, size))
	return scipy.misc.imsave(path, image)

def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	if (images.shape[3] in (3,4)):
		c = images.shape[3]
		img = np.zeros((h * size[0], w * size[1], c))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w, :] = image
		return img
	elif images.shape[3]==1:
		img = np.zeros((h * size[0], w * size[1]))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
		return img
	else:
		raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')

def generate_animation(path, num):
	images = []
	for e in range(num):
		img_name = path + '_epoch%03d' % (e+1) + '.png'
		images.append(imageio.imread(img_name))
	imageio.mimsave(path + '_generate_animation.gif', images, fps=5)

def loss_plot(hist, path='.', model_name='model', y_max=None, use_subplot=False, keys_to_show=[] ):
	try:
		x = range(len(hist['D_loss']))
	except:
		keys = hist.keys()
		lens = [ len(hist[k]) for k in keys if 'loss' in k ]
		maxlen = max(lens)
		x = range(maxlen)

	if use_subplot:
		f, axarr = plt.subplots(2, sharex=True)
		
	plt.xlabel('Iter')
	plt.ylabel('Loss')
	plt.tight_layout()

	if len(keys_to_show) == 0:
		keys_to_show = hist.keys()
	for key,value in hist.items():#hist.iteritems():
		if 'time' in key or key not in keys_to_show:
			continue
		y = value
		if len(x) != len(y):
			print('[warning] loss_plot() found mismatching dimensions: {}'.format(key))
			continue
		if use_subplot and 'acc' in key:
			axarr[1].plot(x, y, label=key)
		elif use_subplot:
			axarr[0].plot(x, y, label=key)
		else:
			plt.plot(x, y, label=key)

	if use_subplot:
		axarr[0].legend(loc=1)
		axarr[0].grid(True)
		axarr[1].legend(loc=1)
		axarr[1].grid(True)
	else:
		plt.legend(loc=1)
		plt.grid(True)


	if y_max is not None:
		if use_subplot:
			x_min, x_max, y_min, _ = axarr[0].axis()
			axarr[0].axis( (x_min, x_max, -y_max/20, y_max) )
		else:
			x_min, x_max, y_min, _ = plt.axis()
			plt.axis( (x_min, x_max, -y_max/20, y_max) )

	path = os.path.join(path, model_name + '_loss.png')

	plt.savefig(path)

	plt.close()

def initialize_weights(net):
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			m.weight.data.normal_(0, 0.02)
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.ConvTranspose2d):
			m.weight.data.normal_(0, 0.02)
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.Conv3d):
			nn.init.xavier_uniform(m.weight)
		elif isinstance(m, nn.ConvTranspose3d):
			nn.init.xavier_uniform(m.weight)
		#elif isinstance(m, nn.GRU):
		#	nn.init.xavier_uniform(m.weight)
		elif isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.02)
			m.bias.data.zero_()


class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()

	def forward(self, x):
		return x.view(x.size(0), -1)


class Inflate(nn.Module):
	def __init__(self, nDims2add):
		super(Inflate, self).__init__()
		self.nDims2add = nDims2add

	def forward(self, x):
		shape = x.size() + (1,)*self.nDims2add
		return x.view(shape)


def parse_args():
	desc = "plot loss"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--fname_hist', type=str, default='', help='history path', required=True)
	parser.add_argument('--fname_dest', type=str, default='.', help='filename of png')
	return parser.parse_args()

if __name__ == '__main__':
	opts = parse_args()
	with open( opts.fname_hist ) as fhandle:
		history = pickle.load(fhandle)
		loss_plot( history, opts.fname_dest )
