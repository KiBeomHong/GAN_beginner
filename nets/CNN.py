import torch, time, os, pickle, math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.misc
from torch.autograd import Variable, grad
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from nets import utils
from nets.utils import Flatten

import pdb


class Net(nn.Module):
	def __init__(self, input_dim):
		super(Net, self).__init__()
		self.input_dim = input_dim

		self.conv = nn.Sequential(

			nn.Conv2d(self.input_dim, 20, 5, 1),
			nn.ReLU(),
			nn.MaxPool2d(2),

			nn.Conv2d(20, 50, 5, 1),
			nn.ReLU(),
			nn.MaxPool2d(2),

		)

		self.fc = nn.Sequential(
			nn.Linear(5*5*50, 500),
			nn.ReLU(),

			nn.Linear(500,10),
			nn.Sigmoid(),
		)
	
		utils.initialize_weights(self)

	def forward(self, input):
		x = self.conv(input)
		x = self.fc(x.view(-1, 5*5*50))
		return x


class CNN(object):
	def __init__(self, args):

		#parameters
		self.batch_size = args.batch_size
		self.epoch = args.epoch
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = args.dataset
		self.dataroot_dir = args.dataroot_dir
		self.model_name = args.model_type + args.comment
		self.sample_num = args.sample_num
		self.gpu_mode = args.gpu_mode
		self.num_workers = args.num_workers
		self.beta1 = args.beta1
		self.beta2 = args.beta2
		self.lr = args.lrG
		self.resl = 32
		self.num_cls = 10


		#load dataset	
		transform = transforms.Compose([transforms.Resize(self.resl),transforms.ToTensor()])

		if not os.path.exists(os.path.join(self.dataroot_dir,self.dataset)):
			os.makedirs(os.path.join(self.dataroot_dir,self.dataset))

		data_path = os.path.join(self.dataroot_dir,self.dataset)

		if self.dataset == 'cifar10':
			dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
			self.data_dim = 3

		elif self.dataset == 'mnist':
			dataset = datasets.MNIST(root=data_path, train=True, download=True, transform=transform)
			self.data_dim = 1

		self.train_loader = DataLoader(dataset, batch_size = self.batch_size, shuffle=True, num_workers = self.num_workers)

		#construct model 
		self.net = Net(self.data_dim)

		#define optimizer
		self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))

		if self.gpu_mode:
			self.net = self.net.cuda()
			self.CE_loss = nn.CrossEntropyLoss().cuda() #BCELoss : Binary Cross Entropy Loss
		else:
			self.CE_loss = nn.CrossEntropyLoss()
	

	def train(self):
		self.train_hist = {}
		self.train_hist['G_loss'] = []
		self.train_hist['per_epoch_time']=[]
		self.train_hist['total_time']=[]
		
		#train
		self.net.train()
		start_time = time.time()

		for epoch in range(self.epoch):
			epoch_start_time = time.time()
			
			for iB, (img_, label_) in enumerate(self.train_loader):
				if iB == self.train_loader.dataset.__len__() // self.batch_size:
					break
					
				if self.gpu_mode:
					label, img_ = Variable(label_.cuda()), Variable(img_.cuda())
				else:
					label, img_ = Variable(label_), Variable(img_)
			
				#----Update cnn_network----#
				self.optimizer.zero_grad()

				output = self.net(img_)
				loss = self.CE_loss(output, label)

				self.train_hist['G_loss'].append(loss.item())
				loss.backward()

				self.optimizer.step()

				#---check train result ----#
				if(iB % 100 == 0) and (epoch%1==0):
					print('[E%03d]'%(epoch)+'  loss :  %.6f '%(loss.item()))
					
			
			#self.visualize_results(epoch, self.z)
			#---check train result ----#
			self.train_hist['per_epoch_time'].append(time.time()-epoch_start_time)
			if not os.path.exists(os.path.join(self.result_dir, self.dataset, self.model_name)):
				os.makedirs(os.path.join(self.result_dir, self.dataset, self.model_name))
			utils.loss_plot(self.train_hist, os.path.join(self.result_dir, self.dataset, self.model_name), self.model_name)
			self.save()
		
		print("Training finish!... save training results")
		self.save()




	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.net.state_dict(), os.path.join(save_dir, self.model_name + '_model.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		self.net.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_model.pkl')))


