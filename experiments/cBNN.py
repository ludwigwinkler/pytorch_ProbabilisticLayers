import torch
import torch.distributions
from torch.distributions import Normal, Uniform
from torch.nn import Sequential, Tanh, ReLU, Linear, Dropout, CELU, BatchNorm1d, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os, argparse, datetime, time, copy, inspect
from argparse import ArgumentParser

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

import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.append("../..")  # Up to -> KFAC -> Optimization -> PHD
cwd = os.path.abspath(os.getcwd())
sys.path.append("/".join(cwd.split("/")[:-3]))
# [print(x) for x in sys.path if 'PhD' in x]

from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Utils import str2bool

from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_DataUtils import MNISTDataModule, FMNISTDataModule
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import BayesLinear, MC_ExpansionLayer
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import MC_CrossEntropyLoss, MC_Accuracy

# print('Python', os.environ['PYTHONPATH'].split(os.pathsep))
# print('MKL', torch.has_mkl)
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.core.memory import ModelSummary

from pl_bolts.datamodules import FashionMNISTDataModule
from pl_bolts.models.autoencoders import VAE

seed_everything(2)


def HParamParser(logger=False,
		 logname='experiment',
		 logdir='experimentdir',
		 dataset=['mnist', 'fmnist'][0],
		 max_epochs=2000,
		 optim=['csgd', 'cbayessgd', 'cvarsgd', 'sgd', 'adam', 'entropy_sgd'][2],
		 model=['nn', 'bnn'][0],
		 plot=True,
		 lr=1.,
		 prior=['1', 'laplace', 'laplace_clamp'][0],
		 verbose=True,
		 ):
	parser = argparse.ArgumentParser()

	# add PROGRAM level args
	parser.add_argument('-name', type=str, default='some_name', help='hi there')
	parser.add_argument('-dataset', type=str, choices=['mnist', 'fmnist'], default=dataset)

	parser.add_argument('-logger', type=str2bool, default=logger)
	parser.add_argument('-logname', type=str, default=logname)
	parser.add_argument('-logdir', type=str, default=logdir)
	parser.add_argument('-plot', type=str2bool, default=plot)
	parser.add_argument('-verbose', type=str2bool, default=verbose)

	parser.add_argument('-optim', type=str, default=optim)
	parser.add_argument('-lr', type=float, default=lr)
	parser.add_argument('-batch_size', type=int, default=80 if optim == 'cvarsgd' else 128)
	parser.add_argument('-max_epochs', type=int, default=max_epochs)

	parser.add_argument('-model', type=str, choices=['nn', 'bnn'], default=model)
	parser.add_argument('-num_MC', type=int, default=13)
	parser.add_argument('-prior', type=str, default=prior)
	parser.add_argument('-num_hidden', type=int, default=200)

	parser.add_argument('-gpus', type=int, default=1 if torch.cuda.is_available() else 0)
	parser.add_argument('-num_workers', type=int, default=0 if torch.cuda.is_available() else 0)

	hparams = parser.parse_args()

	return hparams


class NN(LightningModule):

	@staticmethod
	def add_model_specific_args(parent_parser):
		'''
		Adds arguments to the already existing argument parser 'parent_parser'
		'''
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--num_hidden', type=int, default=200)
		return parser

	def __init__(self, in_features=None, num_classes=None, num_samples=0, **kwargs):

		super().__init__()
		self.save_hyperparameters()

		actfunc = torch.nn.LeakyReLU
		self.nn = Sequential(
					Linear(in_features,self.hparams.num_hidden-1),
				     	actfunc(),
				     	Linear(self.hparams.num_hidden-1,self.hparams.num_hidden+1),
				     	actfunc(),
				     	Linear(self.hparams.num_hidden+1,self.hparams.num_hidden+2),
				     	actfunc(),
				     	Linear(self.hparams.num_hidden+2,num_classes)
				     	)

		self.criterion = CrossEntropyLoss()
		self.summary = ModelSummary(model=self)

	def forward(self, x):

		out = self.nn(x)

		return out

	def training_step(self, batch, batch_idx):
		x, y = batch

		pred = self.forward(x.flatten(1,-1))
		ACC = accuracy(pred, y)
		LL = self.criterion(pred, y)

		progress_bar = {'ACC': ACC}

		if self.hparams.optim == 'csgd':
			u = self.trainer.optimizers[0].collect_u()
			self.log('Min U', np.around(np.min(u),3), prog_bar=True)
			self.log('Med U', np.around(np.median(u),3), prog_bar = True)
			self.log('Max U', np.around(np.max(u),3), prog_bar=True)
			# progress_bar.update({'Min|Median|Max[u]': f'{np.min(u):.3f}|{np.median(u):.3f}|{np.max(u):.3f}'})
			# logs.update({'train/u_min': np.min(u), 'train/u_median': np.median(u), 'train/u_max': np.max(u)})

		# if torch.cuda.is_available():
		# 	progress_bar.update({'GPU MEM': f'{torch.cuda.memory_allocated() / 1024 ** 2 :.3f}'})

		self.log('Train/ACC', ACC, prog_bar=True)
		self.log('Train/Loss', LL, prog_bar=True)

		# return {'loss': LL, 'ACC': ACC, 'progress_bar': progress_bar}
		return {'loss': LL, 'Train/Loss': LL, 'Train/ACC': ACC}

	def training_epoch_end(self, outputs):

		Loss = torch.stack([x['Train/Loss'] for x in outputs]).mean()
		ACC = torch.stack([x['Train/ACC'] for x in outputs]).mean()

		self.log('Train/Epoch_ACC', ACC, prog_bar=True)
		self.log('Train/Epoch_Loss', Loss, prog_bar=True)

	def validation_step(self, batch, batch_idx):
		x, y = batch

		pred = self.forward(x.flatten(1, -1))
		ACC = accuracy(pred, y)
		# ACC = accuracy(pred, y)
		LL = self.criterion(pred, y)

		# return {'val_loss': LL, 'Val_LL': LL, 'Val_ACC': ACC, 'progress_bar': progress_bar}
		return {'Val/Loss': LL, 'Val/LL':LL, 'Val/ACC': ACC}

	def validation_epoch_end(self, outputs):

		LL = torch.stack([x['Val/LL'] for x in outputs]).mean()
		ACC = torch.stack([x['Val/ACC'] for x in outputs]).mean()

		self.log('Val/ACC', ACC, prog_bar=True)
		self.log('Val/LL', LL, prog_bar=True)

	def configure_optimizers(self):

		if self.hparams.optim == 'adam':
			optim = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
		elif self.hparams.optim == 'sgd':
			optim = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, nesterov=False)
		print(str(optim))

		return optim

class BNN(LightningModule):

	@staticmethod
	def add_model_specific_args(parent_parser):
		'''
		Adds arguments to the already existing argument parser 'parent_parser'
		'''
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--num_MC', type=int, default=10)
		parser.add_argument('--num_hidden', type=int, default=200)
		return parser

	def __init__(self, in_features=None, num_classes=None, num_samples=0, **kwargs):

		super().__init__()
		self.save_hyperparameters()

		actfunc = torch.nn.LeakyReLU
		prior = [1., 'laplace'][1]
		self.bnn = Sequential(	MC_ExpansionLayer(num_MC=self.hparams.num_MC, input_dim=2),
					BayesLinear(in_features,self.hparams.num_hidden-1, prior=prior),
				     	actfunc(),
				     	BayesLinear(self.hparams.num_hidden-1,self.hparams.num_hidden+1,prior=prior),
				     	actfunc(),
				     	BayesLinear(self.hparams.num_hidden+1,self.hparams.num_hidden+2, prior=prior),
				     	actfunc(),
				     	BayesLinear(self.hparams.num_hidden+2,num_classes, prior=prior)
				     	)

		self.criterion = MC_CrossEntropyLoss(num_samples=self.hparams.num_samples)

		self.summary = ModelSummary(model=self)
		self.num_params = ModelSummary(model=self).param_nums[0]

		# print(f"{self.summary.param_nums[0]=}")

		# exit(f"Exited @ {inspect.currentframe().f_code.co_name}")

	def forward(self, x):

		batch_size = x.shape[0]

		out = self.bnn(x)

		assert out.shape[:2] == torch.Size([self.hparams.num_MC, batch_size]), f'{out.size()[:2]=} != [{self.hparams.num_MC}, {batch_size}]'

		return out

	def collect_kl_div(self):

		self.kl_div = Tensor([0.0])

		for name, module in self.named_modules():

			if any([isinstance(module, layer) for layer in [BayesLinear]]):
				if hasattr(module, 'kl_div'):
					self.kl_div = self.kl_div + module.kl_div


		return self.kl_div

	def training_step(self, batch, batch_idx):
		x, y = batch

		pred = self.forward(x.flatten(1,-1))
		ACC = MC_Accuracy(pred, y)
		LL = self.criterion(pred, y) / self.hparams.num_samples # per sample loss
		KL = self.collect_kl_div() / self.hparams.num_samples # per sample loss
		H = self.collect_entropy()

		progress_bar = {'ACC': ACC}
		logs = {'train/Loss': LL, 'train/LL': LL, 'train/KL': KL, 'train/ACC': ACC}

		if self.hparams.optim == 'cbayessgd' or self.hparams.optim == 'cvarsgd':
			u = self.trainer.optimizers[0].collect_u()
			self.log('Min U', np.around(np.min(u), 3), prog_bar=True)
			self.log('Med U', np.around(np.median(u), 3), prog_bar=True)
			self.log('Max U', np.around(np.max(u), 3), prog_bar=True)

		# if torch.cuda.is_available():
		# 	progress_bar.update({'GPU MEM': f'{torch.cuda.memory_allocated() / 1024 ** 2 :.3f}'})


		self.log('Train/ACC', ACC, prog_bar=True)
		self.log('Train/Loss', LL+KL, prog_bar=True)
		self.log('Train/LL', LL, prog_bar=True)
		self.log('Train/KL', KL, prog_bar=True)

		output = {'loss': LL+KL, 'LL': LL, 'KL': KL, 'ACC': ACC}

		return output

	def training_epoch_end(self, outputs):

		loss = torch.stack([x['loss'] for x in outputs]).mean()
		LL = torch.stack([x['LL'] for x in outputs]).mean()
		KL = torch.stack([x['KL'] for x in outputs]).mean()
		ACC = torch.stack([x['ACC'] for x in outputs]).mean()

		self.log('Train/Epoch_ACC', ACC, prog_bar=True)
		self.log('Train/Epoch_LL', LL, prog_bar=True)
		self.log('Train/Epoch_KL', KL, prog_bar=True)
		self.log('Train/Epoch_Loss', loss, prog_bar=True)

	def validation_step(self, batch, batch_idx):
		x, y = batch

		pred = self.forward(x.flatten(1, -1))
		ACC = MC_Accuracy(pred, y)
		# ACC = accuracy(pred, y)
		LL = self.criterion(pred, y) / self.hparams.num_samples
		KL = self.collect_kl_div() / self.hparams.num_samples

		return {'Val/LL': LL, 'Val/KL': KL, 'Val/ACC': ACC}

	def validation_epoch_end(self, outputs):

		LL 		= torch.stack([x['Val/LL'] for x in outputs]).mean()
		KL 		= torch.stack([x['Val/KL'] for x in outputs]).mean()
		ACC 		= torch.stack([x['Val/ACC'] for x in outputs]).mean()

		self.log('Val/ACC', ACC, prog_bar=True)
		self.log('Val/LL', LL, prog_bar=True)
		self.log('Val/KL', KL, prog_bar=True)
		self.log('Val/Loss', LL+KL, prog_bar=True)


	def configure_optimizers(self):

		assert self.hparams.optim in ['sgd', 'adam', 'cbayessgd', 'cvarsgd', 'entropy_sgd'], f'{self.hparams.optim=} not a valid optimizer ...'

		if self.hparams.optim == 'adam':
			optim = torch.optim.Adam(self.parameters(), self.hparams.lr)
		elif self.hparams.optim == 'sgd':
			optim = torch.optim.SGD(self.parameters(), lr=self.hparams.lr, nesterov=False)
		print(str(optim))

		return optim

hparams = HParamParser(logger=False, lr=0.001, optim='adam', dataset='mnist', model='nn', prior='laplace_clamp')

if hparams.logger is True:
	# machine = 'cluster' if torch.cuda.is_available() else 'local'
	# if hparams.model=='bnn': logdir = 'logs/'+machine+'/bnn'
	# if hparams.model=='nn': logdir = 'logs/'+machine+'/nn'
	# hparams.__dict__.update({'logname': hparams.optim, 'logdir': logdir})
	from pytorch_lightning.loggers import WandbLogger
	os.system('wandb login --relogin afc4755e33dfa171a8419620e141ebeaeb8f27f5')
	logger = WandbLogger(project='BayesianGradients', entity='ludwigwinkler',name=f'lightning1.0_logging_{hparams.model}')
	hparams.__dict__.update({'logger': logger})

if hparams.dataset=='mnist': dm = MNISTDataModule(hparams)
elif hparams.dataset=='fmnist': dm = FMNISTDataModule(hparams)

if hparams.model=='nn': model = NN(**vars(hparams), in_features=dm.in_features, num_classes=dm.num_classes, num_samples=dm.num_samples)
elif hparams.model=='bnn': model = BNN(**vars(hparams), in_features=dm.in_features, num_classes=dm.num_classes, num_samples=dm.num_samples)

early_stop_callback = EarlyStopping(
	monitor='Val/LL',
	min_delta=0.0,
	patience=1,
	verbose=True,
	mode='min'
)

trainer = Trainer.from_argparse_args(	hparams,
					progress_bar_refresh_rate=5,
					min_steps=1000,
					# row_log_interval=1,
					# gradient_clip_val=1. if hparams.optim in 'sgd' else 0.,
				  	callbacks=[early_stop_callback],
					# limit_train_batches=20,
					checkpoint_callback=False,
				  	)

trainer.fit(model, datamodule=dm)