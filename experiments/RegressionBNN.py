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
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import BayesLinear, MC_ExpansionLayer
from pytorch_ProbabilisticLayers.data.ProbabilisticLayers_SyntheticData import generate_nonstationary_data

from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Utils import RunningAverageMeter
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Loss import MC_NLL
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Utils import MC_GradientCorrection

# print('Python', os.environ['PYTHONPATH'].split(os.pathsep))
# print('MKL', torch.has_mkl)
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
seed_everything(2)



class LitBayesNN(LightningModule):

	@staticmethod
	def add_model_specific_args(parent_parser):
		'''
		Adds arguments to the already existing argument parser 'parent_parser'
		'''
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--num_MC', type=int, default=20)
		parser.add_argument('--in_features', type=int, default=1)
		parser.add_argument('--out_features', type=int, default=1)
		parser.add_argument('--num_hidden', type=int, default=50)
		parser.add_argument('--batch_size', type=int, default=50)
		return parser

	def __init__(self, **kwargs):

		super().__init__()
		self.save_hyperparameters()

		actfunc = torch.nn.ReLU

		self.nn = Sequential(MC_ExpansionLayer(num_MC=self.hparams.num_MC, input_dim=2),
				     BayesLinear(1, self.hparams.num_hidden - 1),
				     actfunc(),
				     BayesLinear(self.hparams.num_hidden - 1, self.hparams.num_hidden + 1),
				     actfunc(),
				     BayesLinear(self.hparams.num_hidden + 1, self.hparams.num_hidden + 2),
				     actfunc(),
				     BayesLinear(self.hparams.num_hidden + 2, 1)
				     )

		# self.mc_gradientcorrection = MC_GradientCorrection(self.parameters(), num_MC=self.hparams.num_MC)
		self.criterion = MC_NLL(num_samples=self.hparams.num_samples)  # mean requires rescaling of gradient

	def forward(self, x):

		batch_size = x.shape[0]

		out = self.nn(x)

		assert out.dim()==3
		assert out.shape[:2] == torch.Size([self.hparams.num_MC, batch_size]), f'{out.size()[:2]=} != [{self.hparams.num_MC}, {batch_size}]'

		return out

	def collect_kl_div(self):

		self.kl_div = Tensor([0.])

		for name, module in self.named_modules():

			if any([isinstance(module, layer) for layer in [BayesLinear]]):
				if hasattr(module, 'kl_div'):
					self.kl_div = self.kl_div + module.kl_div

		return self.kl_div

	def training_step(self, batch, batch_idx):
		x, y = batch

		pred = self.forward(x)
		MSE = F.mse_loss(pred.mean(dim=0), y)
		LL = self.criterion(pred, y) / self.hparams.num_samples
		KL = self.collect_kl_div() / self.hparams.num_samples

		self.log('Train/LL', LL, prog_bar=True)
		self.log('Train/KL', KL, prog_bar=True)
		self.log('Train/MSE', MSE, prog_bar=True)
		return {'loss': LL+KL}
		# return {'loss': MSE, 'progress_bar': progress_bar}

	def validation_step(self, batch, batch_idx):
		x, y = batch
		pred = self.forward(x)

		MSE = F.mse_loss(pred.mean(dim=0), y)
		LL = self.criterion(pred, y) / self.hparams.num_samples
		KL = self.collect_kl_div() / self.hparams.num_samples

		return {'Val/Loss': LL + KL, 'Val/LL': LL, 'Val/KL': KL, 'Val/MSE': MSE}
		# return {'val_loss': MSE, 'Val_NLL': NLL, 'Val_KL': KL, 'Val_MSE': MSE, 'progress_bar': progress_bar}

	def validation_epoch_end(self, outputs):

		val_loss = torch.stack([x['Val/Loss'] for x in outputs]).mean()
		LL = torch.stack([x['Val/LL'] for x in outputs]).mean()
		KL = torch.stack([x['Val/KL'] for x in outputs]).mean()
		MSE = torch.stack([x['Val/MSE'] for x in outputs]).mean()

		self.log('Val/Loss', LL+KL, prog_bar=True)
		self.log('Val/LL', LL, prog_bar=True)
		self.log('Val/KL', KL, prog_bar=True)
		self.log('Val/MSE', MSE, prog_bar=True)

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

litbnn = LitBayesNN(**vars(args), pi=np.pi)

early_stop_callback = EarlyStopping(
	monitor='Val/Loss',
	min_delta=0.0,
	patience=10,
	verbose=True,
	mode='min'
)

trainer = Trainer.from_argparse_args(	args,
					progress_bar_refresh_rate=5,
					# min_steps=1000,
				  	# limit_train_batches=0.3,
				  	callbacks=[early_stop_callback],
				  	check_val_every_n_epoch=1,
				  	logger=False,
				  	)

trainer.fit(litbnn, train_dataloader=train_loader, val_dataloaders=val_loader)