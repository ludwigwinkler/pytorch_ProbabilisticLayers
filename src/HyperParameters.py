import argparse
import numpy as np
import matplotlib, warnings
import torch
from Utils.Utils import str2bool

matplotlib.rcParams["figure.figsize"] = [10, 10]

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

from pytorch_lightning.loggers import TensorBoardLogger


def HParamParser(logger=False,
		 entity='ludwigwinkler',
		 project='arandomexperiment',
		 dataset=['mnist', 'fmnist', 'cifar10'][0],
		 max_epochs=2000,
		 fast_dev_run=False,
		 optim=['csgd', 'bayescsgd', 'stochcontrolsgd', 'sgd', 'adam', 'entropy_sgd'][2],
		 model=['cnn', 'nn', 'bnn'][0],
		 plot=True,
		 lr = 0.001,
		 num_MC=5,
		 batch_size=128,
		 prior=['1', 'laplace', 'laplace_clamp'][0],
		 verbose=True,
		 ):

	parser = argparse.ArgumentParser()

	# add PROGRAM level args
	parser.add_argument('--logger', '-logger', type=str2bool, default=logger)
	parser.add_argument('--entity', type=str, default=entity)
	parser.add_argument('--project', type=str, default=project)
	parser.add_argument('--experiment', type=str, default=None, help='hi there')

	parser.add_argument('--dataset', type=str, choices=['mnist', 'fmnist', 'cifar10'], default=dataset)
	parser.add_argument('--plot', type=str2bool, default=plot)
	parser.add_argument('--verbose', type=str2bool, default=verbose)
	parser.add_argument('--fast_dev_run', type=str2bool, default=fast_dev_run)

	parser.add_argument('--optim', type=str, default=optim)
	parser.add_argument('--lr', '-lr', type=float, default=lr)
	parser.add_argument('--max_epochs', type=int, default=max_epochs)

	parser.add_argument('--batch_size', '-batch_size', type=int, default=batch_size)

	parser.add_argument('--model', type=str, choices=['nn', 'cnn', 'bnn', 'cbnn', 'resnet18'], default=model)
	parser.add_argument('--num_MC', '-num_MC', type=int, default=num_MC)
	parser.add_argument('--prior', type=str, default=prior)
	parser.add_argument('--num_hidden', type=int, default=200)
	parser.add_argument('--num_channels', type=int, default=100)

	parser.add_argument('--entropy_sgd_langeviniters', type=int, default=100)

	parser.add_argument('--gpus', type=int, default=1 if torch.cuda.is_available() else 0)
	parser.add_argument('--num_workers', type=int, default=4 if torch.cuda.device_count()>1 else 0)

	hparams = parser.parse_args()

	hparams.__dict__.update({'experiment': f"{hparams.model}_{hparams.dataset}_{hparams.optim}" if hparams.experiment is None else f"{hparams.experiment}_{hparams.model}_{hparams.dataset}_{hparams.optim}"})

	assert hparams.optim in ['sgd', 'adam', 'csgd', 'bayescsgd', 'stochcontrolsgd', 'entropy_sgd'], f"{hparams.optim} not a valid optimizer"
	assert hparams.model in ['nn', 'bnn', 'cnn', 'cbnn', 'resnet18'], f"{hparams.model} not a valid optimizer"
	# Catching model & optimizer pairs
	if hparams.model in ['nn', 'cnn', 'resnet18']:
		assert hparams.optim in ['csgd', 'sgd', 'adam', 'entropy_sgd'], f"Can't use {hparams.optim} with {hparams.model}"
	elif hparams.model in ['bnn', 'cbnn']:
		assert hparams.optim in ['bayescsgd', 'stochcontrolsgd', 'sgd', 'adam', 'entropy_sgd'], f"Can't use {hparams.optim} with {hparams.model}"

	if torch.cuda.device_count()>1: # for more gpus, scale batch_size and num_workers linearly for each gpu
		hparams.batch_size = hparams.batch_size*torch.cuda.device_count()
		hparams.num_workers = hparams.num_workers*torch.cuda.device_count()

	return hparams