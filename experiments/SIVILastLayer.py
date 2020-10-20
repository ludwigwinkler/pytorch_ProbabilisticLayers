import torch
import torch.distributions
from torch.distributions import Normal, Uniform
from torch.nn import Sequential, Tanh, ReLU, Linear, Dropout, CELU, BatchNorm1d
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, os, argparse, datetime, time, copy
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

# sys.path.append("..")  # Up to -> KFAC -> Optimization -> PHD
# [print(x) for x in sys.path if 'PhD' in x]
cwd = os.path.abspath(os.getcwd())
sys.path.append("/".join(cwd.split("/")[:-2]))

from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import BayesianNeuralNetwork
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import BayesLinear, VariationalNormal, MC_BatchNorm1D, SIVILayer
from pytorch_ProbabilisticLayers.data.ProbabilisticLayers_SyntheticData import generate_nonstationary_data

from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Utils import RunningAverageMeter
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Loss import MC_NLL
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Utils import MC_GradientCorrection

# print('Python', os.environ['PYTHONPATH'].split(os.pathsep))
# print('MKL', torch.has_mkl)
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
# seed_everything(2)



class LitBayesNN(LightningModule):

	@staticmethod
	def add_model_specific_args(parent_parser):
		'''
		Adds arguments to the already existing argument parser 'parent_parser'
		'''
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--num_MC', type=int, default=100)
		parser.add_argument('--in_features', type=int, default=1)
		parser.add_argument('--out_features', type=int, default=1)
		parser.add_argument('--num_hidden', type=int, default=50)
		parser.add_argument('--batch_size', type=int, default=100)
		return parser

	def __init__(self, **kwargs):

		super().__init__()
		self.save_hyperparameters()

		self.fc1 = BayesLinear(self.hparams.in_features,
				       self.hparams.num_hidden,
				       self.hparams.num_MC, prior_scale=1.)
		self.fc2 = BayesLinear(self.hparams.num_hidden,
				       self.hparams.num_hidden,
				       self.hparams.num_MC, prior_scale=1.)
		self.fc3 = BayesLinear(self.hparams.num_hidden,
				       self.hparams.out_features,
				       num_MC=self.hparams.num_MC, prior_scale=1.)

		# self.mc_gradientcorrection = MC_GradientCorrection(self.parameters(), num_MC=self.hparams.num_MC)
		self.criterion = MC_NLL(num_samples=self.hparams.num_samples)  # mean requires rescaling of gradient

	def forward(self, x):

		batch_size = x.shape[0]
		out = x.unsqueeze(0).repeat(self.hparams.num_MC, *(x.dim() * (1,)))
		assert out.dim()==3

		actfunc = F.leaky_relu

		out = self.fc1(out)
		out = actfunc(out)
		out = self.fc2(out)
		out = actfunc(out)
		out = self.fc3(out)

		assert out.shape[:2] == torch.Size([self.hparams.num_MC, batch_size]), f'{out.size()[:2]=} != [{self.hparams.num_MC}, {batch_size}]'

		return out

	def collect_kl_div(self):

		self.kl_div = 0

		for name, module in self.named_children():

			if any([isinstance(module, layer) for layer in [BayesLinear]]):
				if hasattr(module, 'kl_div'):
					self.kl_div = self.kl_div + module.kl_div

		return self.kl_div

	def training_step(self, batch, batch_idx):
		x, y = batch

		pred = self.forward(x)
		MSE = F.mse_loss(pred.mean(dim=0), y)
		NLL = self.criterion(pred, y) / self.hparams.num_samples
		KL = self.collect_kl_div() / self.hparams.num_samples

		progress_bar = {'NLL': NLL, 'KL': KL, 'MSE': MSE}
		return {'loss': NLL+KL, 'progress_bar': progress_bar}

	def validation_step(self, batch, batch_idx):
		x, y = batch
		pred = self.forward(x)

		MSE = F.mse_loss(pred.mean(dim=0), y)
		NLL = self.criterion(pred, y) / self.hparams.num_samples
		KL = self.collect_kl_div() / self.hparams.num_samples

		progress_bar = {'Val_NLL': NLL, 'Val_KL': KL, 'Val_MSE': MSE}
		return {'val_loss': NLL + KL, 'Val_NLL': NLL, 'Val_KL': KL, 'Val_MSE': MSE, 'progress_bar': progress_bar}

	def validation_epoch_end(self, outputs):

		val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		NLL = torch.stack([x['Val_NLL'] for x in outputs]).mean()
		KL = torch.stack([x['Val_KL'] for x in outputs]).mean()
		MSE = torch.stack([x['Val_MSE'] for x in outputs]).mean()

		# if (self.trainer.current_epoch+1)%(self.trainer.max_epochs//10)==0:
		# 	self.plot_prediction()

		progress_bar = {'val_loss': val_loss, 'Val_NLL': NLL, 'Val_KL': KL, 'Val_MSE': MSE}
		return {'val_loss': val_loss, 'progress_bar': progress_bar}

	def on_fit_end(self):

		self.plot_prediction()
		# a=1

	def plot_prediction(self):

		with torch.no_grad():

				x_pred = torch.linspace(3*x.min(), 3*x.max(), 100).reshape(-1,1)
				pred = self.forward(x_pred)

				x_pred = x_pred.squeeze()


				mu = pred.mean(dim=0).squeeze()
				std = pred.std(dim=0).squeeze()


				fig = plt.figure()
				plt.xlim(3*x.min(), 3*x.max())
				plt.ylim(2*y.min(), 2*y.max())

				plt.scatter(x,y, alpha=0.25, s=1)
				plt.plot(x_pred, mu, color='red', alpha=0.5)

				plt.fill_between(x_pred, mu-1*std, mu+1*std, alpha=0.25, color='red')
				plt.fill_between(x_pred, mu-2*std, mu+2*std, alpha=0.10, color='red')
				plt.fill_between(x_pred, mu-3*std, mu+3*std, alpha=0.05, color='red')
				plt.title(f"Epoch: {self.trainer.current_epoch}")
				plt.grid()
				plt.show()

	def configure_optimizers(self):

		return torch.optim.Adam(self.parameters(), lr=0.01)
		# return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, nesterov=True)

	def backward(self, trainer, loss: Tensor, optimizer: torch.optim.Optimizer, optimizer_idx: int) -> None:

		loss.backward()

		# for name, param in model.named_parameters():
		# 	print('*'*100)
		# 	print(name)
		# 	print(".grad.shape:             ", param.grad.shape)
		# 	print(".grad_batch.shape:       ", param.grad_batch.shape)
		#
		# exit()

	def optimizer_step(self,
			   epoch: int,
			   batch_idx: int,
			   optimizer: torch.optim.Optimizer,
			   optimizer_idx: int,
			   second_order_closure=None,
			   on_tpu: bool = False,
			   using_native_amp: bool = False,
			   using_lbfgs: bool = False):

		# self.mc_gradientcorrection.step()
		optimizer.step()

		# self.trainer.progress_bar_metrics.update({'dμ': copy.deepcopy(self.fc1.weight.loc.grad[0,0])/self.fc1.sampled_w[:,0,0].mean().detach()})
		# self.trainer.progress_bar_metrics.update({'dLdμ': copy.deepcopy(self.fc1.weight.loc.grad[0, 0])})
		# self.trainer.progress_bar_metrics.update({'locs.grad.mean': copy.deepcopy(self.fc3.locs.grad[:,0,0].mean())})
		# self.trainer.progress_bar_metrics.update({'Std[dLdμ]': copy.deepcopy(self.fc1.locs.grad[:, 0, 0].std())})
		#
		# self.trainer.progress_bar_metrics.update({'dρ': copy.deepcopy(self.fc1.weight.logscale.grad[0,0]) / self.fc1.sampled_w[:,0, 0].mean().detach()})
		# self.trainer.progress_bar_metrics.update({'dLdρ': copy.deepcopy(self.fc1.weight.logscale.grad[0, 0])})
		# self.trainer.progress_bar_metrics.update({'Std[dLdρ]': copy.deepcopy(self.fc1.logscales.grad[:, 0, 0].std())})
		# self.trainer.progress_bar_metrics.update({'E[dLdρ]': copy.deepcopy(self.fc1.logscales.grad[:, 0, 0].mean())})
		# print(f"{self.fc1.logscales.grad[:20,0,0]=}")
		optimizer.zero_grad()

class LastLayerSIVI(LightningModule):

	@staticmethod
	def add_model_specific_args(parent_parser):
		'''
		Adds arguments to the already existing argument parser 'parent_parser'
		'''
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--num_MC', type=int, default=25)
		parser.add_argument('--in_features', type=int, default=1)
		parser.add_argument('--out_features', type=int, default=1)
		parser.add_argument('--num_hidden', type=int, default=50)
		parser.add_argument('--batch_size', type=int, default=100)
		return parser

	def __init__(self, **kwargs):

		super().__init__()
		self.save_hyperparameters()

		self.fc1 = Linear(self.hparams.in_features, self.hparams.num_hidden)
		self.fc2 = Linear(self.hparams.num_hidden, self.hparams.num_hidden)
		self.fc3 = SIVILayer(self.hparams.num_hidden,
				       self.hparams.out_features,
				       _dim_noise_input=5)
		# self.fc3 = BayesLinear(self.hparams.num_hidden,
		# 		       self.hparams.out_features)

		# self.mc_gradientcorrection = MC_GradientCorrection(self.parameters(), num_MC=self.hparams.num_MC)
		self.criterion = MC_NLL(num_samples=self.hparams.num_samples)  # mean requires rescaling of gradient

	def forward(self, x):

		batch_size = x.shape[0]
		# out = x.unsqueeze(0).repeat(self.hparams.num_MC, *(x.dim() * (1,)))r
		# assert out.dim()==3

		actfunc = [torch.tanh, F.celu, F.leaky_relu][2]

		out = self.fc1(x)
		out = actfunc(out)
		out = self.fc2(out)
		out = actfunc(out)
		# print(f"{out.shape=}")
		out = out.unsqueeze(0).repeat(self.hparams.num_MC, 1, 1)
		out = self.fc3(out)

		assert out.shape[:2] == torch.Size([self.hparams.num_MC, batch_size]), f'{out.size()[:2]=} != [{self.hparams.num_MC}, {batch_size}]'

		return out

	def collect_kl_div(self):

		self.kl_div = 0.

		for name, module in self.named_children():

			if any([isinstance(module, layer) for layer in [BayesLinear, SIVILayer]]):
				if hasattr(module, 'kl_div'):
					self.kl_div = self.kl_div + module.kl_div

		return self.kl_div

	def training_step(self, batch, batch_idx):
		x, y = batch

		pred = self.forward(x)
		MSE = F.mse_loss(pred.mean(dim=0), y)
		NLL = self.criterion(pred, y) / self.hparams.num_samples
		KL = self.collect_kl_div() / self.hparams.num_samples

		progress_bar = {'NLL': NLL, 'KL': KL, 'MSE': MSE}
		return {'loss': NLL+KL, 'progress_bar': progress_bar}

	def validation_step(self, batch, batch_idx):
		x, y = batch
		pred = self.forward(x)

		MSE = F.mse_loss(pred.mean(dim=0), y)
		NLL = self.criterion(pred, y) / self.hparams.num_samples
		KL = self.collect_kl_div() / self.hparams.num_samples

		progress_bar = {'Val_NLL': NLL, 'Val_KL': KL, 'Val_MSE': MSE}
		return {'val_loss': NLL + KL, 'Val_NLL': NLL, 'Val_KL': KL, 'Val_MSE': MSE, 'progress_bar': progress_bar}

	def validation_epoch_end(self, outputs):

		val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		NLL = torch.stack([x['Val_NLL'] for x in outputs]).mean()
		KL = torch.stack([x['Val_KL'] for x in outputs]).mean()
		MSE = torch.stack([x['Val_MSE'] for x in outputs]).mean()

		# if (self.trainer.current_epoch+1)%(self.trainer.max_epochs//10)==0:
		# 	self.plot_prediction()

		progress_bar = {'val_loss': val_loss, 'Val_NLL': NLL, 'Val_KL': KL, 'Val_MSE': MSE}
		return {'val_loss': val_loss, 'progress_bar': progress_bar}

	def on_fit_end(self):

		self.plot_prediction()
		self.fc3.sample_posterior_dist()

	def plot_prediction(self):

		with torch.no_grad():

				x_pred = torch.linspace(3*x.min(), 3*x.max(), 100).reshape(-1,1)
				pred = self.forward(x_pred)

				x_pred = x_pred.squeeze()


				mu = pred.mean(dim=0).squeeze()
				std = pred.std(dim=0).squeeze()


				fig = plt.figure()
				plt.xlim(3*x.min(), 3*x.max())
				plt.ylim(2*y.min(), 2*y.max())

				plt.scatter(x,y, alpha=0.25, s=1)
				plt.plot(x_pred, mu, color='red', alpha=0.5)

				plt.fill_between(x_pred, mu-1*std, mu+1*std, alpha=0.25, color='red')
				plt.fill_between(x_pred, mu-2*std, mu+2*std, alpha=0.10, color='red')
				plt.fill_between(x_pred, mu-3*std, mu+3*std, alpha=0.05, color='red')
				plt.title(f"Epoch: {self.trainer.current_epoch}")
				plt.grid()
				plt.show()

	def configure_optimizers(self):

		return torch.optim.Adam(self.parameters(), lr=0.01)
		# return torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9, nesterov=True)

	def backward(self, trainer, loss: Tensor, optimizer: torch.optim.Optimizer, optimizer_idx: int) -> None:

		loss.backward()

		# for name, param in model.named_parameters():
		# 	print('*'*100)
		# 	print(name)
		# 	print(".grad.shape:             ", param.grad.shape)
		# 	print(".grad_batch.shape:       ", param.grad_batch.shape)
		#
		# exit()

	def optimizer_step(self,
			   epoch: int,
			   batch_idx: int,
			   optimizer: torch.optim.Optimizer,
			   optimizer_idx: int,
			   second_order_closure=None,
			   on_tpu: bool = False,
			   using_native_amp: bool = False,
			   using_lbfgs: bool = False):

		# self.mc_gradientcorrection.step()
		optimizer.step()

		# self.trainer.progress_bar_metrics.update({'dμ': copy.deepcopy(self.fc1.weight.loc.grad[0,0])/self.fc1.sampled_w[:,0,0].mean().detach()})
		# self.trainer.progress_bar_metrics.update({'dLdμ': copy.deepcopy(self.fc1.weight.loc.grad[0, 0])})
		# self.trainer.progress_bar_metrics.update({'locs.grad.mean': copy.deepcopy(self.fc3.locs.grad[:,0,0].mean())})
		# self.trainer.progress_bar_metrics.update({'Std[dLdμ]': copy.deepcopy(self.fc1.locs.grad[:, 0, 0].std())})
		#
		# self.trainer.progress_bar_metrics.update({'dρ': copy.deepcopy(self.fc1.weight.logscale.grad[0,0]) / self.fc1.sampled_w[:,0, 0].mean().detach()})
		# self.trainer.progress_bar_metrics.update({'dLdρ': copy.deepcopy(self.fc1.weight.logscale.grad[0, 0])})
		# self.trainer.progress_bar_metrics.update({'Std[dLdρ]': copy.deepcopy(self.fc1.logscales.grad[:, 0, 0].std())})
		# self.trainer.progress_bar_metrics.update({'E[dLdρ]': copy.deepcopy(self.fc1.logscales.grad[:, 0, 0].mean())})
		# print(f"{self.fc1.logscales.grad[:20,0,0]=}")
		optimizer.zero_grad()


parser = argparse.ArgumentParser()

parser.add_argument('--num_samples', type=int, default=1000)
parser.add_argument('--x_noise_std', type=float, default=0.01)
parser.add_argument('--y_noise_std', type=float, default=.2)

parser.add_argument('--max_epochs', type=int, default=1000)
parser.add_argument('--progress_bar_refresh_rate', type=int, default=5)
parser = LitBayesNN.add_model_specific_args(parser)
args = parser.parse_args()

x, y = generate_nonstationary_data(num_samples=args.num_samples,
				   x_noise_std=args.x_noise_std,
				   y_noise_std=args.y_noise_std,
				   plot=False)

x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-1)
y = (y - y.mean(dim=0)) / (y.std(dim=0) + 1e-1)
train_loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x, y), batch_size=args.num_samples, shuffle=False)

# litbnn = LitBayesNN(**vars(args), pi=np.pi)
litbnn = LastLayerSIVI(**vars(args))

early_stop_callback = EarlyStopping(
	min_delta=0.00,
	patience=10,
	verbose=True,
	mode='min'
)

# trainer = Trainer(args, progress_bar_refresh_rate=1,
# 		  # limit_train_batches=0.3,
# 		  early_stop_callback=early_stop_callback,
# 		  check_val_every_n_epoch=5,
# 		  # row_log_interval=10,
# 		  logger=False,
# 		  )
trainer = Trainer.from_argparse_args(	args,
					progress_bar_refresh_rate=1,
				  	# limit_train_batches=0.3,
				  	early_stop_callback=early_stop_callback,
				  	check_val_every_n_epoch=5,
				  	logger=False,
				  	)

trainer.fit(litbnn, train_dataloader=train_loader, val_dataloaders=val_loader)