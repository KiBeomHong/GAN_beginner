import torch, time, os, pickle, imageio, math
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

class Gen(nn.Module):
	def __init__(self):
		super(Gen, self).__init__()
		self.input_dim = 100
		self.output_dim = 3

		self.fc = nn.Sequential(
			nn.Linear(self.input_dim, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 128*16*16),
			nn.BatchNorm1d(128*16*16),
			nn.ReLU(),
		)

		self.deconv = nn.Sequential(
			#16*16 -> 8*8 -> 16*16
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(128, 64, 4, 2, 1, bias=False),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			#16 -> 32
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(64, 32, 3, 1, 1, bias=False),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			#32 -> 64
			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(32, self.output_dim, 3, 1, 1, bias=False),
			nn.Sigmoid(),

		)
	
		utils.initialize_weights(self)

	def forward(self, z):
		x = self.fc(z)
		x = x.view(-1, 128, 16, 16)
		x = self.deconv(x)
		
		return x

class Dis(nn.Module):
	def __init__(self):
		super(Dis, self).__init__()
		self.input_dim = 3
		self.output_dim  = 1

		self.conv = nn.Sequential(
			nn.Conv2d(self.input_dim, 64, 4 ,2 ,1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(64, 128, 4, 2, 1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2),
		)

		self.fc = nn.Sequential(
			nn.Linear(128*16*16, 1024),
			nn.BatchNorm1d(1024),
			nn.LeakyReLU(0.2),
			nn.Linear(1024, self.output_dim),
			nn.Sigmoid(),
		)
	
	def forward(self, input):
		x = self.conv(input)
		x = x.view(-1, 128*16*16)
		x = self.fc(x)

		return x



class WGAN(object):
	def __init__(self, args):
		#parameters
		self.batch_size = args.batch_size
		self.epoch = args.epoch
		self.save_dir = args.save_dir
		self.result_dir = args.result_dir
		self.dataset = args.dataset
		self.dataroot_Img_dir = args.dataroot_dir
		self.model_name = args.gan_type + args.comment
		self.sample_num = args.sample_num
		self.gpu_mode = args.gpu_mode
		self.num_workers = args.num_workers
		self.beta1 = args.beta1
		self.beta2 = args.beta2
		self.lrG = args.lrG
		self.lrD = args.lrD
		self.resl = 64
		self.use_gp = True # True : WGAN_GP / False: WGAN


		#load dataset	
		self.train_loader = DataLoader(utils.CUB(root_dir = '/media/sunny/b98f42f1-1582-4e30-bfd4-d7f27abe9ea1/data/CUB/CUB_200_2011/CUB_200_2011/images/',transform=transforms.Compose([transforms.CenterCrop(160), transforms.Scale(64), transforms.ToTensor()])), batch_size = self.batch_size, shuffle=True)

		#construct model G & D
		self.G = Gen()
		self.D = Dis()

		#define optimizer for G & D
		self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
		self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))

		if self.gpu_mode:
			self.G = self.G.cuda()
			self.D = self.D.cuda()
			self.BCE_loss = nn.BCELoss().cuda() #BCELoss : Binary Cross Entropy Loss
		else:
			self.BEC_loss = nn.BECLoss()

	def train(self):
		self.train_hist = {}
		self.train_hist['G_loss'] = []
		self.train_hist['D_loss'] = []
		self.train_hist['per_epoch_time']=[]
		self.train_hist['total_time']=[]

		if self.gpu_mode:
			self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1).cuda()), Variable(torch.zeros(self.batch_size, 1).cuda())
			self.z = Variable(torch.rand(self.batch_size, 100).cuda())
		else:
			self.y_real_, self.y_fake_ = Variable(torch.ones(self.batch_size, 1)), Variable(torch.zeros(self.batch_size, 1))
			self.z = Variable(torch.rand(self.batch_size,100))
		

		#train
		self.D.train()
		start_time = time.time()

		for epoch in range(self.epoch):
			self.G.train()
			epoch_start_time = time.time()
			for iB, img_ in enumerate(self.train_loader):
				if iB == self.train_loader.dataset.__len__() // self.batch_size:
					break

				#Latent space
				z_ = torch.rand(self.batch_size, 100)
					
				if self.gpu_mode:
					z_, img_ = Variable(z_.cuda()), Variable(img_.cuda())	
				else:
					z_, img_ = Variable(z_), Variable(img_)
			

				#----Update D_network----#
				self.D_optimizer.zero_grad()

				D_real = self.D(img_)
				D_real_loss = -torch.mean(D_real)
			
				G_ = self.G(z_)
				D_fake = self.D(G_)
				D_fake_loss = torch.mean(D_fake)

				#gradient penalty
				if self.use_gp:
					if self.gpu_mode:
						alpha = torch.rand(img_.size()).cuda()
					else:
						alpha = torch.rand(img.size())
					x_hat = Variable(alpha*img_.data + (1-alpha)*G_.data, requires_grad=True)
					pred_hat = self.D(x_hat)
					if self.gpu_mode:
						gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()).cuda(),create_graph=True, retain_graph=True, only_inputs=True)[0]
					else:
						gradients = grad(outputs=pred_hat, inputs=x_hat, grad_outputs=torch.ones(pred_hat.size()), create_grahp=True, retain_graph=True, only_inputs=True)[0]
					
					gradient_penalty = self.lambda_*((gradients.contiguous().view(gradients.size()[0], -1).norm(2,1) -1)**2).mean()
					
					D_loss = D_fake_loss + D_real_loss + gradient_penalty

				else:
					D_loss = D_fake_loss + D_real_loss

				self.train_hist['D_loss'].append(D_loss.data[0])
				self.D_optimizer.step()
				

				#---Update G_network---#
				self.G_optimizer.zero_grad()	
				G_ = self.G(z_)
				D_fake = self.D(G_)
				G_fake_loss = -torch.mean(D_fake)
				G_loss = G_fake_loss
				self.train_hist['G_loss'].append(G_loss.data[0])
				G_loss.backward()
				self.G_optimizer.step()

				#---check train result ----#
				if(iB % 100 == 0) and (epoch%10==0):
					print('[E%03d]'%(epoch)+'  G_loss :  %.6f '%(G_loss.data[0])+'  D_loss :  %.6f = %.6f + %.6f'%(D_loss.data[0], D_fake_loss.data[0], D_real_loss.data[0])+'   D_acc : %.6f'%D_acc)
					self.visualize_results(epoch, z_, img_, iB)
					self.G.train()

			#---check train result ----#
			self.train_hist['per_epoch_time'].append(time.time()-epoch_start_time)
			utils.loss_plot(self.train_hist, os.path.join(self.result_dir, self.dataset, self.model_name), self.model_name)
			self.save()
		
		print("Training finish!... save training results")
		self.save()



	def visualize_results(self, epoch, z_, y, iB, fix=True):
		self.G.eval()
		if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.model_name):
			os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.model_name)

		tot_num_samples = min(self.sample_num, self.batch_size)
		image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

		if fix:
			""" fixed noise """
			samples = self.G(z_)
		
		if self.gpu_mode:
			samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
			gt = y.cpu().data.numpy().transpose(0, 2, 3, 1)
		else:
			samples = samples.data.numpy().transpose(0, 2, 3, 1)
			gt = y.data.numpy().transpose(0, 2, 3, 1)

		utils.save_images(samples[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim], self.result_dir+'/'+self.dataset+'/'+self.model_name+'/'+self.model_name+'_epoch%03d'%epoch+'_I%03d'%iB+'_F.png')
		utils.save_images(gt[:image_frame_dim*image_frame_dim,:,:,:], [image_frame_dim, image_frame_dim], self.result_dir+'/'+self.dataset+'/'+self.model_name+'/'+self.model_name+'_epoch%03d'%epoch+'_I%03d'%iB+'_R.png')


	def save(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

		torch.save(self.G.state_dict(), os.path.join(save_dir, self.model_name + '_G.pkl'))
		torch.save(self.D.state_dict(), os.path.join(save_dir, self.model_name + '_D.pkl'))
		#torch.save(self.GRU.state_dict(), os.path.join(save_dir, self.model_name + '_GRU.pkl'))

		with open(os.path.join(save_dir, self.model_name + '_history.pkl'), 'wb') as f:
			pickle.dump(self.train_hist, f)

	def load(self):
		save_dir = os.path.join(self.save_dir, self.dataset, self.model_name)

		self.G.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_G.pkl')))
		self.D.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_D.pkl')))
		self.GRU.load_state_dict(torch.load(os.path.join(save_dir, self.model_name + '_GRU.pkl')))


