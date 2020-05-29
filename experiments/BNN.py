import torch
import torch.distributions
from torch.distributions import Normal, Uniform
from torch.nn import Sequential, Tanh, ReLU, Linear, Dropout, CELU, BatchNorm1d
from torch.utils.tensorboard import SummaryWriter


import sklearn
from sklearn.datasets import make_moons
from sklearn.preprocessing import scale

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy
import sys, os, argparse, datetime, time

# matplotlib.use('svg')
matplotlib.rcParams['figure.figsize'] = (10,10)
plt.rcParams['svg.fonttype'] = 'none'
# exit()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# FloatTensor = torch.FloatTensor
if torch.cuda.is_available():
	FloatTensor = torch.cuda.FloatTensor
	Tensor = torch.cuda.FloatTensor
elif not torch.cuda.is_available():
	FloatTensor = torch.FloatTensor
	Tensor = torch.FloatTensor

import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import BayesianNeuralNetwork
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import BayesLinear, VariationalNormal, MC_BatchNorm1D
from pytorch_ProbabilisticLayers.data.ProbabilisticLayers_SyntheticData import generate_nonstationary_data

from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Utils import RunningAverageMeter
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Loss import MC_NLL

# print('Python', os.environ['PYTHONPATH'].split(os.pathsep))
# print('MKL', torch.has_mkl)

params = argparse.ArgumentParser()
params.add_argument('-logname',                   type=str,           default=' BNNCluster')
params.add_argument('-logging',                   type=int,           default=0)

params.add_argument('-num_hidden',                type=int,           default=50)
params.add_argument('-num_samples',               type=int,           default=500)
params.add_argument('-x_noise_std',               type=float,         default=0.01)
params.add_argument('-y_noise_std',               type=float,         default=.5)
params.add_argument('-zoom',                      type=int,           default=3)

params.add_argument('-lr',                        type=float,         default=0.005)

params.add_argument('-num_epochs',                type=int,           default=200)
params.add_argument('-batch_size',                type=int,           default=250)
params.add_argument('-num_MC',                    type=int,           default=25)
params.add_argument('-val_patience',              type=int,           default=200)

params = params.parse_args()



class BayesNN(BayesianNeuralNetwork):

	def __init__(self, in_features, out_features, num_MC):

		BayesianNeuralNetwork.__init__(self, num_MC=num_MC)

		self.fc1 = BayesLinear(in_features, params.num_hidden, num_MC=num_MC)
		self.fc2 = BayesLinear(params.num_hidden, params.num_hidden, num_MC=num_MC)
		# self.fc3 = BayesLinear(params.num_hidden, params.num_hidden, num_MC=num_MC)
		self.fc4 = BayesLinear(params.num_hidden, out_features, num_MC=num_MC)

	def forward(self, x):

		out = super().forward(x)

		actfunc = F.leaky_relu_

		out = self.fc1(out)
		out = actfunc(out)
		out = self.fc2(out)
		out = actfunc(out)
		# x = self.fc3(x)
		# x = actfunc(x)
		out = self.fc4(out)

		assert out.shape[0]==self.num_MC

		return out

class BayesNN_(torch.nn.Module):

	def __init__(self, num_MC=11, prior_scale=1.):
		super().__init__()

		self.num_MC = num_MC
		self.stochastic = True
		self.prior = True
		self.prior_scale = prior_scale

		self.bn0            = MC_BatchNorm1D(num_features=1)
		self.fc1            = BayesLinear(1, params.num_hidden, num_MC=num_MC)
		self.bn1            = MC_BatchNorm1D(num_features=params.num_hidden)
		self.fc2            = BayesLinear(params.num_hidden, params.num_hidden, num_MC=num_MC)
		self.fc3            = BayesLinear(params.num_hidden, params.num_hidden, num_MC=num_MC)
		self.fc4            = BayesLinear(params.num_hidden, 1, num_MC=num_MC)

	def forward(self, x, num_MC=None):

		out = x.unsqueeze(0).repeat((self.num_MC, 1, 1)) if num_MC is None else x.unsqueeze(0).repeat((num_MC, 1, 1))

		prior = Tensor([0.]) if self.prior else None

		# out = self.bn0(out)
		# out = (out - out.mean(dim=[0,1]))/(out.std(dim=[0,1])+1e-3)
		# print(f'{out.mean(dim=[0,1])=} {out.std(dim=[0,1])=}')
		# exit()
		out, prior = self.fc1(out, prior=prior, stochastic=self.stochastic)
		# out = self.bn1(out)
		out = F.celu_(out)
		out, prior = self.fc2(out, prior=prior, stochastic=self.stochastic)
		out = F.celu_(out)
		# out, prior = self.fc3(out, prior=prior, stochastic=self.stochastic)
		# out = F.relu(out)
		out, prior = self.fc4(out, prior=prior, stochastic=self.stochastic)

		prior = prior if self.prior else Tensor([0.])

		assert prior.shape==torch.Size([1])

		return out, prior.item()


x, y = generate_nonstationary_data(x_noise_std=params.x_noise_std, y_noise_std=params.y_noise_std, plot=False)


x = (x-x.mean(dim=0))/(x.std(dim=0)+1e-1)
# x +=3
y = (y-y.mean(dim=0))/(y.std(dim=0)+1e-1)
dataloader = DataLoader(TensorDataset(x, y), batch_size=params.batch_size, shuffle=True, drop_last=False)

model_type ='bnn'

if model_type=='bnn':
	model = BayesNN(in_features=x.shape[-1], out_features=y.shape[-1], num_MC=params.num_MC)
elif model_type=='nn':
	model = Sequential(Linear(1, params.num_hidden), torch.nn.ReLU(), Linear(params.num_hidden, params.num_hidden), torch.nn.ReLU(),
			 Linear(params.num_hidden, 1))

optim = torch.optim.RMSprop(model.parameters(), lr=params.lr)

if model_type=='bnn':
	criterion = MC_NLL(reduction='sum')
	mse_criterion = lambda pred, target: F.mse_loss(pred, target.unsqueeze(0).repeat(pred.shape[0],1,1))
elif model_type=='nn':
	criterion = lambda pred, target: F.mse_loss(pred, target)
	mse_criterion = lambda pred, target: F.mse_loss(pred, target)


loss = RunningAverageMeter(0.99)
mse_loss = RunningAverageMeter(0.99)
if model_type=='bnn':
	kl_div = RunningAverageMeter(0.99)
	pred_uncertainty = RunningAverageMeter(0.99)

# print(f"{len(dataloader)=}")
# exit()

gif_frames = []
for epoch in range(params.num_epochs):

	# bnn.train()

	progress = tqdm(enumerate(dataloader)) if epoch%(params.num_epochs//10)==0 else enumerate(dataloader)
	for batch_i, (batch_data, batch_target) in progress:

		optim.zero_grad()

		batch_pred = model(batch_data)
		batch_loss = criterion(batch_pred, batch_target)

		if model_type == 'bnn':
			batch_kl_div = model.collect_kl_div() / len(dataloader)

		loss.update(batch_loss.detach().item())
		mse_loss.update(mse_criterion(batch_pred, batch_target).detach().item())

		if model_type == 'bnn':
			kl_div.update(batch_kl_div.detach().item())
			pred_uncertainty.update(batch_pred.std(dim=0).mean().detach().item())
			batch_loss = batch_loss + batch_kl_div

		batch_loss.backward()
		optim.step()

		if model_type=='bnn':
			desc= f'Epoch: {epoch} \t Loss: {loss.avg:.2f} \t KL: {kl_div.avg:.2f} \t MSE: {mse_loss.avg:.2f} \t σ: {pred_uncertainty.avg:.2f}'
		elif model_type=='nn':
			desc = f'Epoch: {epoch} \t Loss: {loss.avg:.2f} \t MSE: {mse_loss.avg:.2f}'
		progress.set_description(desc) if epoch%(params.num_epochs//10)==0 else None

	# if epoch % (params.num_epochs // 10) == 0 and epoch>=0:
	if epoch % 5 == 0 and epoch>=0:
	# if epoch==0:
		with torch.no_grad():

			if model_type=='bnn':
				model.num_MC = 100

			x_pred = torch.linspace(3*x.min(), 3*x.max(), 100).reshape(-1,1)
			pred = model(x_pred)

			x_pred = x_pred.squeeze()

			if model_type == 'bnn':
				mu = pred.mean(dim=0).squeeze()
				std = pred.std(dim=0).squeeze()
			elif model_type =='nn':
				mu = pred.squeeze()

			fig = plt.figure()
			plt.xlim(3*x.min(), 3*x.max())
			plt.ylim(2*y.min(), 2*y.max())
			# plt.xticks([])
			# plt.yticks([])

			# [plt.plot(x_pred, pred_, alpha=0.1, c='red') for pred_ in pred]

			plt.scatter(x,y, alpha=0.25, s=1)
			plt.plot(x_pred, mu, color='red', alpha=0.5)
			if model_type=='bnn':
				plt.fill_between(x_pred, mu-1*std, mu+1*std, alpha=0.25, color='red')
				plt.fill_between(x_pred, mu-2*std, mu+2*std, alpha=0.10, color='red')
				plt.fill_between(x_pred, mu-3*std, mu+3*std, alpha=0.05, color='red')
			plt.title(f"Epoch: {epoch}")
			plt.grid()
			# plt.show()

			fig.canvas.draw()  # draw the canvas, cache the renderer
			image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
			image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

			gif_frames.append(image)

import imageio
imageio.mimsave('BNN.gif', gif_frames, fps=2)
