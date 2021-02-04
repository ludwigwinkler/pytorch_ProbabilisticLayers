import torch
import torch.distributions
from torch.distributions import Normal, Uniform
from torch.nn import Sequential, Flatten
from torch.nn import Linear, Conv2d, Dropout, BatchNorm2d, CrossEntropyLoss, MaxPool2d
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os, argparse, datetime, time, copy, inspect
from argparse import ArgumentParser
from math import prod

from typing import Optional, Callable
from torch.optim import Optimizer

# matplotlib.use('svg')
matplotlib.rcParams['figure.figsize'] = (10,10)
plt.rcParams['svg.fonttype'] = 'none'
# exit()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
	FloatTensor = torch.cuda.FloatTensor
	Tensor = torch.cuda.FloatTensor
elif not torch.cuda.is_available():
	FloatTensor = torch.FloatTensor
	Tensor = torch.FloatTensor

# sys.path.append("..")  # Up to -> KFAC -> Optimization -> PHD
cwd = os.path.abspath(os.getcwd())
sys.path.append("/".join(cwd.split("/")[:-1]))
sys.path.append("/".join(cwd.split("/")[:-2]))
sys.path.append("/".join(cwd.split("/")[:-3]))
# [print(x) for x in sys.path if 'PhD' in x]

# exit()


from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import BayesLinear, MC_ExpansionLayer, BayesConv2d, MC_MaxPool2d, MC_BatchNorm2D
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import MC_CrossEntropyLoss, MC_Accuracy, BayesAdaptiveInit_FlattenAndLinear
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_DataUtils import MNISTDataModule, FMNISTDataModule, CIFAR10DataModule

from pytorch_ProbabilisticLayers.src.HyperParameters import HParamParser
from pytorch_ProbabilisticLayers.src.ModelUtils import AdaptiveInit_FlattenAndLinear, CIFAR10_ReferenceNet, CIFAR10_ResNet18, PrintModule, ResNet18


# print('Python', os.environ['PYTHONPATH'].split(os.pathsep))
# print('MKL', torch.has_mkl)
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.core.memory import ModelSummary

# seed_everything(123)

class NN(LightningModule):

	# @staticmethod
	# def add_model_specific_args(parent_parser):
	# 	'''
	# 	Adds arguments to the already existing argument parser 'parent_parser'
	# 	'''
	# 	parser = ArgumentParser(parents=[parent_parser], add_help=False)
	# 	parser.add_argument('--num_hidden', type=int, default=200)
	# 	return parser

	def __init__(self, in_dims=None, num_classes=None, **kwargs):

		super().__init__()
		self.save_hyperparameters()

		actfunc = torch.nn.LeakyReLU
		if self.hparams.model=='nn':
			in_features = prod(in_dims)
			self.nn = Sequential(	Flatten(1,-1),
						Linear(in_features,self.hparams.num_hidden),
						actfunc(),
						Linear(self.hparams.num_hidden,self.hparams.num_hidden),
						actfunc(),
						Linear(self.hparams.num_hidden,self.hparams.num_hidden),
						actfunc(),
						# Linear(self.hparams.num_hidden,self.hparams.num_hidden),
						# actfunc(),
						# Linear(self.hparams.num_hidden,self.hparams.num_hidden),
						# actfunc(),
						Linear(self.hparams.num_hidden,num_classes)
						)
		elif self.hparams.model=='cnn':
			layer_args = {'kernel_size':5, 'padding':2, 'stride':1}
			self.nn = Sequential(
						Conv2d(in_channels=in_dims[0], out_channels=96, **layer_args),
						BatchNorm2d(96),
						actfunc(),
						MaxPool2d(kernel_size=2, stride=2),
						Conv2d(in_channels=96, out_channels=128, **layer_args),
						BatchNorm2d(128),
						actfunc(),
						MaxPool2d(kernel_size=2, stride=2),
						Conv2d(in_channels=128, out_channels=256, **layer_args),
						BatchNorm2d(256),
						actfunc(),
						MaxPool2d(kernel_size=2, stride=2),
						Conv2d(in_channels=256, out_channels=128, **layer_args),
						BatchNorm2d(128),
						actfunc(),
						MaxPool2d(kernel_size=2, stride=2),
						AdaptiveInit_FlattenAndLinear(self.hparams.num_hidden),
						actfunc(),
						Linear(self.hparams.num_hidden, num_classes)
					)
			self.nn(torch.randn(1, *in_dims))

		elif self.hparams.model=='resnet18':
			self.nn = ResNet18(in_dims[0])

		self.criterion = CrossEntropyLoss()
		self.summary = ModelSummary(model=self)

	def forward(self, x):
		out = self.nn(x)
		return out

	def training_step(self, batch, batch_idx):
		x, y = batch

		pred = self.nn(x)
		ACC = accuracy(pred, y)
		LL = self.criterion(pred, y)
		loss = LL
		LL, ACC = LL.detach(), ACC.detach()

		if self.hparams.optim == 'csgd':
			u = self.trainer.optimizers[0].collect_u()
			self.log_dict({'MinU': np.around(np.min(u), 3),
				       'MedU': np.around(np.median(u), 3),
				       'MaxU': np.around(np.max(u), 3)},
				      	prog_bar=True)

		if torch.cuda.is_available():
			self.log_dict({'GPU MEM': np.around(torch.cuda.memory_allocated()/1024**2,3)}, prog_bar=True)

		self.log_dict({'Train/ACC': ACC}, prog_bar=True)
		self.log_dict({'Train/Loss': LL}, prog_bar=False)
		return {"loss": loss, "Train/Loss": LL, "Train/ACC": ACC}

	def training_epoch_end(self, outputs):

		Loss = torch.stack([x['Train/Loss'] for x in outputs]).mean()
		ACC = torch.stack([x['Train/ACC'] for x in outputs]).mean()

		self.log_dict({"Train/EpochACC": ACC}, prog_bar=True)
		self.log_dict({"Train/EpochLoss": Loss}, prog_bar=False)

	def validation_step(self, batch, batch_idx):
		x, y = batch

		pred = self.forward(x)
		ACC = accuracy(pred, y)
		LL = self.criterion(pred, y)

		return {"Val/Loss": LL, "Val/LL":LL, "Val/ACC": ACC}

	def validation_epoch_end(self, outputs):

		LL = torch.stack([x['Val/LL'] for x in outputs]).mean()
		ACC = torch.stack([x['Val/ACC'] for x in outputs]).mean()

		# print(f"Val/EpochACC {self.current_epoch}: {ACC}")

		self.log_dict({'Val/EpochACC': ACC}, prog_bar=True)
		self.log_dict({'Val/EpochLoss': LL, 'Val/EpochLL': LL}, prog_bar=False)

	def configure_optimizers(self):

		if self.hparams.optim == 'adam':
			optim = torch.optim.Adam(self.nn.parameters(), lr=self.hparams.lr)
			# return [optim], [torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=200)]
			return optim
		elif self.hparams.optim == 'sgd':
			optim = torch.optim.SGD(self.nn.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay = 5e-4)
			scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=20, eta_min=0.0001)
			# return [optim], [scheduler]
			return optim
		else:
			raise ValueError("Unknown optimizer")

class BNN(LightningModule):

	# @staticmethod
	# def add_model_specific_args(parent_parser):
	# 	'''
	# 	Adds arguments to the already existing argument parser 'parent_parser'
	# 	'''
	# 	parser = ArgumentParser(parents=[parent_parser], add_help=False)
	# 	parser.add_argument('--num_MC', type=int, default=10)
	# 	parser.add_argument('--num_hidden', type=int, default=200)
	# 	return parser

	def __init__(self, in_dims=None, num_classes=None, num_samples=0, **kwargs):

		super().__init__()
		self.save_hyperparameters()

		actfunc = torch.nn.LeakyReLU
		bias = True
		prior = [1., 'laplace'][0]
		if self.hparams.model == 'bnn':
			in_features = prod(in_dims)
			self.bnn = Sequential(	Flatten(1, -1),
						MC_ExpansionLayer(num_MC=self.hparams.num_MC, input_dim=2),
						BayesLinear(in_features,self.hparams.num_hidden, prior=prior, bias=bias),
						actfunc(),
						BayesLinear(self.hparams.num_hidden,self.hparams.num_hidden,prior=prior, bias=bias),
						actfunc(),
						BayesLinear(self.hparams.num_hidden,self.hparams.num_hidden, prior=prior, bias=bias),
						actfunc(),
						# BayesLinear(self.hparams.num_hidden,self.hparams.num_hidden, prior=prior, bias=bias),
						# actfunc(),
						# BayesLinear(self.hparams.num_hidden,self.hparams.num_hidden, prior=prior, bias=bias),
						# actfunc(),
						BayesLinear(self.hparams.num_hidden,num_classes, prior=prior, bias=bias)
						# BayesLinear(self.hparams.num_hidden + 1,num_classes, prior=prior)
						)

		if self.hparams.model == 'cbnn':
			debug = 1
			layer_args = {'kernel_size': 5, 'padding': 2, 'stride': 1, 'num_MC': self.hparams.num_MC}
			self.bnn = Sequential(	MC_ExpansionLayer(num_MC=self.hparams.num_MC, input_dim=4),
						# PrintModule(),
						BayesConv2d(in_channels=in_dims[0], out_channels=int(96/debug), **layer_args),
						MC_BatchNorm2D(int(96/debug)),
						actfunc(),
						MC_MaxPool2d(kernel_size=2, stride=2),
						BayesConv2d(in_channels=int(96/debug), out_channels=int(128/debug), **layer_args),
						MC_BatchNorm2D(int(128/debug)),
						actfunc(),
						MC_MaxPool2d(kernel_size=2, stride=2),
						BayesConv2d(in_channels=int(128/debug), out_channels=int(256/debug), **layer_args),
						MC_BatchNorm2D(int(256/debug)),
						actfunc(),
						MC_MaxPool2d(kernel_size=2, stride=2),
						BayesConv2d(in_channels=int(256/debug), out_channels=int(128/debug), **layer_args),
						MC_BatchNorm2D(int(128/debug)),
						actfunc(),
						MC_MaxPool2d(kernel_size=2, stride=2),
						BayesAdaptiveInit_FlattenAndLinear(self.hparams.num_hidden),
						actfunc(),
						BayesLinear(self.hparams.num_hidden, num_classes)
			)
			self.bnn(torch.randn(1, *in_dims, dtype=torch.float32))

		self.criterion = MC_CrossEntropyLoss(num_samples=self.hparams.num_samples)

		self.summary = ModelSummary(model=self)
		self.num_params = ModelSummary(model=self).param_nums[0]


	def forward(self, x):

		batch_size = x.shape[0]
		out = self.bnn(x)

		assert out.shape[:2] == torch.Size([self.hparams.num_MC, batch_size]), f"{out.size()[:2]=} != [{self.hparams.num_MC}, {batch_size}]"
		return out

	def collect_kl_div(self):

		self.kl_div = Tensor([0.0])
		for name, module in self.named_modules():
			if any([isinstance(module, layer) for layer in [BayesLinear]]):
				if hasattr(module, 'kl_div'):
					self.kl_div = self.kl_div + module.kl_div
		return self.kl_div

	def collect_entropy(self):

		self.entropy = Tensor([0.0])
		for name, module in self.named_modules():
			if any([isinstance(module, layer) for layer in [BayesLinear]]):
				if hasattr(module, 'entropy'):
					self.entropy = self.entropy + module.entropy
		return self.entropy

	def training_step(self, batch, batch_idx):
		x, y = batch

		pred = self.forward(x)
		ACC = MC_Accuracy(pred, y)
		LL = self.criterion(pred, y) / self.hparams.num_samples # per sample loss
		KL = self.collect_kl_div() / self.hparams.num_samples # per sample loss
		H = self.collect_entropy()

		loss = LL+KL
		LL, KL, H = LL.detach(), KL.detach(), H.detach()

		if self.hparams.optim == 'bayescsgd' or self.hparams.optim == 'stochcontrolsgd':
			u = self.trainer.optimizers[0].collect_u()
			self.log_dict({'MinU': np.around(np.min(u), 3),
				       'MedU': np.around(np.median(u), 3),
				       'MaxU': np.around(np.max(u), 3)},
				      prog_bar=True)

		if torch.cuda.is_available():
			gpu_mem = np.around((torch.cuda.memory_allocated() / 1024 ** 2), 3)
			self.log_dict({'GPU MEM': gpu_mem}, prog_bar=True)

		self.log_dict({'Train/ACC': ACC}, prog_bar=True)
		self.log_dict({'Train/Loss': LL+KL, 'Train/LL': LL, 'Train/KL': KL}, prog_bar=False)

		return {'loss': loss, 'Train/ACC': ACC, 'Train/Loss': LL+KL, 'Train/LL': LL, 'Train/KL': KL}

	def training_epoch_end(self, outputs):

		loss = torch.stack([x['loss'] for x in outputs]).mean()
		LL = torch.stack([x['Train/LL'] for x in outputs]).mean()
		KL = torch.stack([x['Train/KL'] for x in outputs]).mean()
		ACC = torch.stack([x['Train/ACC'] for x in outputs]).mean()

		self.log_dict({'Train/EpochACC': ACC}, prog_bar=True)
		self.log_dict({'Train/EpochLoss': loss, 'Train/EpochLL': LL, 'Train/EpochKL': KL}, prog_bar=False)

	def validation_step(self, batch, batch_idx):
		x, y = batch

		pred = self.forward(x)
		ACC = MC_Accuracy(pred, y)
		LL = self.criterion(pred, y) / self.hparams.num_samples
		KL = self.collect_kl_div() / self.hparams.num_samples

		return {'Val/Loss': LL+KL, 'Val/LL': LL, 'Val/KL': KL, 'Val/ACC': ACC}

	def validation_epoch_end(self, outputs):

		LL 		= torch.stack([x['Val/LL'] for x in outputs]).mean()
		KL 		= torch.stack([x['Val/KL'] for x in outputs]).mean()
		ACC 		= torch.stack([x['Val/ACC'] for x in outputs]).mean()

		self.log_dict({'Val/EpochACC': ACC}, prog_bar=True)
		self.log_dict({'Val/EpochLoss': LL+KL, 'Val/EpochLL': LL, 'Val/EpochKL': KL}, prog_bar=False)

	def configure_optimizers(self):

		assert self.hparams.optim in ['sgd', 'adam', 'bayescsgd', 'stochcontrolsgd', 'entropy_sgd'], f'{self.hparams.optim=} not a valid optimizer for {self.hparams.model}'

		if self.hparams.optim == 'adam':
			optim = torch.optim.Adam(self.parameters(), self.hparams.lr)
		elif self.hparams.optim == 'sgd':
			optim = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, nesterov=False)
		else:
			raise ValueError("Unknown optimizer")

		return optim

hparams = HParamParser(logger=True, project='BNN_num_MC', entity='ludwigwinkler',
		       dataset='fmnist', model='bnn',
		       lr=0.001, optim='adam', num_MC=1, batch_size=128)

print()
[print(f"{key}: {value}") for key, value in vars(hparams).items() if key in ['model', 'lr', 'optim', 'dataset', 'batch_size']]
print()

if hparams.logger is True:
	from pytorch_lightning.loggers import WandbLogger
	# os.system('wandb login --relogin afc4755e33dfa171a8419620e141ebeaeb8f27f5')
	logger = WandbLogger(project=hparams.project, entity=hparams.entity,name=hparams.experiment)
	hparams.__dict__.update({'logger': logger})

print(f"{hparams.num_workers=}")

if hparams.dataset=='mnist': dm = MNISTDataModule(hparams)
elif hparams.dataset=='fmnist': dm = FMNISTDataModule(hparams)
# elif hparams.dataset=='cifar10': dm = CIFAR10DataModule(batch_size=hparams.batch_size, num_workers=hparams.num_workers)

if hparams.model in ['cnn', 'nn', 'resnet18']: model = NN(**vars(hparams), in_dims=dm.dims, num_classes=dm.num_classes, num_samples=dm.num_samples)
elif hparams.model in ['bnn', 'cbnn']: model = BNN(**vars(hparams), in_dims=dm.dims, num_classes=dm.num_classes, num_samples=dm.num_samples)

early_stop_callback = EarlyStopping(	monitor="Val/EpochACC",
					min_delta=0.01, # 0.8213 -> 82.13%
					patience=1,
					verbose=True,
					mode='max'
					)


trainer = Trainer.from_argparse_args(	hparams,
					# log_every_n_steps=1,
					progress_bar_refresh_rate=5,
					# min_epochs=200,
					max_epochs=500,
					gradient_clip_val=1. if hparams.optim in ['sgd'] else 0.,
					# auto_scale_batch_size='power' if torch.cuda.device_count()>0 else None,
					# auto_scale_batch_size='power',
					# fast_dev_run = hparams.fast_dev_run,
					checkpoint_callback=False,
				  	callbacks=[early_stop_callback],
					gpus=torch.cuda.device_count() if torch.cuda.device_count()>0 else None,
					distributed_backend="ddp" if torch.cuda.device_count()>1 else None
				  	)

trainer.fit(model, datamodule=dm)