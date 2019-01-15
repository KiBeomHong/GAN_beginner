import argparse, os, pickle
from nets.GAN import GAN
from nets.WGAN import WGAN
from nets.DiracGAN import DiracGAN
from nets.CNN import CNN

import pdb

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

"""parsing and configuration"""
def parse_args():
	desc = "Pytorch implementation of GAN collections"
	parser = argparse.ArgumentParser(description=desc)

	parser.add_argument('--model_type', type=str, default='GAN', choices=['GAN', 'WGAN','DiracGAN','CNN','RNN'], help='The type of GAN')#, required=True)
	parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist','cifar10'], help='The name of dataset')
	parser.add_argument('--dataroot_dir', type=str, default='./data', help='root path of data')
	parser.add_argument('--epoch', type=int, default=25, help='The number of epochs to run')
	parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
	parser.add_argument('--sample_num', type=int, default=64, help='The number of samples to test')
	parser.add_argument('--save_dir', type=str, default='./models', help='Directory name to save the model')
	parser.add_argument('--result_dir', type=str, default='./results', help='Directory name to save the generated images')
	parser.add_argument('--lrG', type=float, default=0.0002)
	parser.add_argument('--lrD', type=float, default=0.0002)
	parser.add_argument('--beta1', type=float, default=0.5)
	parser.add_argument('--beta2', type=float, default=0.999)
	parser.add_argument('--gpu_mode', type=str2bool, default=True)
	parser.add_argument('--num_workers', type=int, default='1', help='number of threads for DataLoader')
	parser.add_argument('--comment', type=str, default='', help='comment to put on model_name')

	# below arguments are for eval mode
	parser.add_argument('--type', type=str, default='train', help='train or test')

	return check_args(parser.parse_args())

"""checking arguments"""
def check_args(opts):
	# --save_dir
	if not os.path.exists(opts.save_dir):
		os.makedirs(opts.save_dir)

	# --result_dir
	if not os.path.exists(opts.result_dir):
		os.makedirs(opts.result_dir)

	# --comment
	if len(opts.comment)>0:
		print( "comment: " + opts.comment )
		comment_part = '_'+opts.comment
	else:
		comment_part = ''
	tempconcat = opts.model_type+comment_part
	print( 'models and loss plot -> ' + os.path.join( opts.save_dir, opts.dataset, tempconcat ) )
	print( 'results -> ' + os.path.join( opts.result_dir, opts.dataset, tempconcat ) )

	# --epoch
	try:
		assert opts.epoch >= 1
	except:
		print('number of epochs must be larger than or equal to one')

	# --batch_size
	try:
		assert opts.batch_size >= 1
	except:
		print('batch size must be larger than or equal to one')

	print( opts )

	return opts

"""main"""
def main():
	#parse arguments
	opts = parse_args()
	if opts is None:
		print("There is no opts!!")		
		exit()

	if opts.model_type == 'CNN':
		model = CNN(opts)
	elif opts.model_type == 'RNN':
		model = RNN(opts)	
	elif opts.model_type == 'GAN':
		model = GAN(opts)
	elif opts.model_type == 'WGAN':
		model = WGAN(opts)
	elif opts.model_type == 'DiracGAN':
		model = WGAN(opts)
	else:
		raise Exception("[!] There is no option for " + opts.model_type)

	if opts.type == 'train':
		model.train()
		print("[*] Training finished")
		
	elif opts.type == 'test':
		model.test()
		print("[*] Test finished")

if __name__ == '__main__':
	main()
