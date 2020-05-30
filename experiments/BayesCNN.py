import torch
import torch.distributions
from torch.distributions import Normal, Uniform
from torch.nn import Sequential, Tanh, ReLU, Linear, Dropout, CELU, BatchNorm1d
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import copy
import sys, os, argparse, datetime, time

# matplotlib.use('svg')
matplotlib.rcParams['figure.figsize'] = (10, 10)
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

sys.path.append("../..")

from torch.nn import Conv2d, Linear, MaxPool2d

from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import BayesianNeuralNetwork
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import BayesLinear, BayesConv2d, MC_BatchNorm1D, MC_MaxPool2d
from pytorch_ProbabilisticLayers.data.ProbabilisticLayers_SyntheticData import generate_nonstationary_data

from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Utils import RunningAverageMeter
from pytorch_ProbabilisticLayers.src.ProbabilisticLayers_Loss import MC_NLL, MC_CrossEntropyLoss, MC_Accuracy

# print('Python', os.environ['PYTHONPATH'].split(os.pathsep))
# print('MKL', torch.has_mkl)

params = argparse.ArgumentParser()
params.add_argument('-logname', type=str, default=' BNNCluster')
params.add_argument('-logging', type=int, default=0)

params.add_argument('-data', 	choices=['mnist', 'fmnist'], default='fmnist')
params.add_argument('-model', 	choices=['bcnn', 'cnn'], default='bcnn')

params.add_argument('-lr', type=float, default=0.0001)

params.add_argument('-num_epochs', type=int, default=200)
params.add_argument('-batch_size', type=int, default=100)
params.add_argument('-num_MC', type=int, default=5)
params.add_argument('-val_patience', type=int, default=200)

params = params.parse_args()

print(params)

class CNN(torch.nn.Module):

	def __init__(self, in_channels, out_classes):
		super().__init__()
		self.in_channels = in_channels
		self.out_classes = out_classes

		conv1_out_maps = 64
		conv2_out_maps = 256

		self.conv1 = Conv2d(in_channels, conv1_out_maps, kernel_size=5)
		self.maxpool1 = MaxPool2d(kernel_size=2)
		self.conv2 = Conv2d(conv1_out_maps, conv2_out_maps, kernel_size=5)
		self.maxpool2 = MaxPool2d(kernel_size=2)
		self.lin1 = Linear(in_features=4096, out_features=1000)
		self.lin2 = Linear(in_features=1000, out_features=500)
		self.lin3 = Linear(in_features=500, out_features=out_classes)

	def forward(self, x):

		out = self.conv1(x)
		out = F.leaky_relu(out)
		out = self.maxpool1(out)

		out = self.conv2(out)
		out = F.leaky_relu(out)
		out = self.maxpool2(out)

		out = out.flatten(-3, -1)

		out = self.lin1(out)
		out = F.leaky_relu(out)

		out = self.lin2(out)
		out = F.leaky_relu(out)

		out = self.lin3(out)

		return out


class BayesCNN(BayesianNeuralNetwork):

	def __init__(self, in_channels, out_classes, num_MC):

		BayesianNeuralNetwork.__init__(self, num_MC=num_MC)

		self.in_channels = in_channels
		self.out_classes = out_classes

		conv1_out_maps = 64
		conv2_out_maps = 256

		self.conv1 = BayesConv2d(in_channels, conv1_out_maps, kernel_size=5, num_MC=num_MC)
		self.maxpool1 = MC_MaxPool2d(kernel_size=2)
		self.conv2 = BayesConv2d(conv1_out_maps, conv2_out_maps, kernel_size=5, num_MC=num_MC)
		self.maxpool2 = MC_MaxPool2d(kernel_size=2)
		self.lin1 = BayesLinear(in_features=4096, out_features=1000, num_MC=num_MC)
		self.lin2 = BayesLinear(in_features=1000, out_features=500, num_MC=num_MC)
		self.lin3 = BayesLinear(in_features=500, out_features=out_classes, num_MC=num_MC)

	def collect_kl_div(self):

		self.kl_div = 0

		for name, module in self.named_children():

			# print(f'@kl_div {any([isinstance(module, layer) for layer in [BayesLinear, BayesConv2d]])}')

			if any([isinstance(module, layer) for layer in [BayesLinear, BayesConv2d]]):
				self.kl_div = self.kl_div + module.kl_div

		return self.kl_div

	def forward(self, x):

		out = super().forward(x)

		out = self.conv1(out)
		out = F.leaky_relu(out)
		out = self.maxpool1(out)

		out = self.conv2(out)
		out = F.leaky_relu(out)
		out = self.maxpool2(out)

		out = out.flatten(-3,-1)

		out = self.lin1(out)
		out = F.leaky_relu(out)

		out = self.lin2(out)
		out = F.leaky_relu(out)

		out = self.lin3(out)

		return out





if params.data=='mnist':
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2885,), (0.3568,))])

	train_data = datasets.MNIST('../../Data/MNIST/',download=True,train=True,transform=transform)
	trainloader = torch.utils.data.DataLoader(train_data,batch_size=params.batch_size,shuffle=True)

	testing = datasets.MNIST('../../Data/MNIST/',download=True,train=False,transform=transform)
	testloader = torch.utils.data.DataLoader(testing,batch_size=params.batch_size,shuffle=True)

elif params.data=='fmnist':
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.2885,), (0.3568,))])

	train_data = datasets.FashionMNIST('../../Data/FMNIST/',download=True,train=True,transform=transform)
	trainloader = torch.utils.data.DataLoader(train_data,batch_size=params.batch_size,shuffle=True)

	testing = datasets.FashionMNIST('../../Data/FMNIST/',download=True,train=False,transform=transform)
	testloader = torch.utils.data.DataLoader(testing,batch_size=params.batch_size,shuffle=True)






if params.model == 'bcnn':
	model = BayesCNN(in_channels=1, out_classes=10, num_MC=params.num_MC).to(device)
	criterion = MC_CrossEntropyLoss()
	mse_criterion = lambda pred, target: F.mse_loss(pred, target.unsqueeze(0).repeat(pred.shape[0], 1, 1))
	kl_div = RunningAverageMeter(0.99)
	pred_uncertainty = RunningAverageMeter(0.99)

elif params.model == 'cnn':
	model = CNN(in_channels=1, out_classes=10).to(device)
	criterion = torch.nn.CrossEntropyLoss()
	mse_criterion = lambda pred, target: F.mse_loss(pred, target.unsqueeze(0).repeat(pred.shape[0], 1, 1))

optim = torch.optim.Adam(model.parameters(), lr=params.lr)

train_loss = RunningAverageMeter(0.99)
train_acc = RunningAverageMeter(0.99, init_avg=0.1234)
train_uncertainty = RunningAverageMeter(0.99, init_avg=0.)

val_loss = RunningAverageMeter(0.9)
val_acc = RunningAverageMeter(0.9, init_avg=0.1234)
val_uncertainty = RunningAverageMeter(0.99, init_avg=0.)



for epoch in range(params.num_epochs):

	# bnn.train()

	progress = tqdm(enumerate(trainloader))
	for batch_i, (batch_data, batch_target) in progress:

		batch_data, batch_target = batch_data.to(device), batch_target.to(device)


		optim.zero_grad()

		batch_pred = model(batch_data)
		batch_loss = criterion(batch_pred, batch_target)

		train_loss.update(batch_loss.detach().item())
		train_acc.update(MC_Accuracy(batch_pred, batch_target))

		if params.model == 'bcnn':
			# batch_kl_div = model.collect_kl_div() / len(trainloader)
			# batch_loss = batch_loss + batch_kl_div
			train_uncertainty.update(batch_pred.std(dim=0).mean().detach().item())

		batch_loss.backward()
		optim.step()

		desc = f"Epoch: {epoch} \t Loss: {train_loss.avg:.3f} \t Acc: {train_acc.avg:.3f} \t σ: {train_uncertainty.avg:.3f} | Loss: {val_loss.avg:.3f} Acc: {val_acc.avg:.3f}"

		if params.model == 'bcnn':
			desc += f' KL: {kl_div.avg:.2f}\t σ:'

		progress.set_description(desc)

	with torch.no_grad():

		for batch_i, (batch_data, batch_target) in enumerate(testloader):

			batch_data, batch_target = batch_data.to(device), batch_target.to(device)

			batch_pred = model(batch_data)

			batch_loss = criterion(batch_pred, batch_target)

			val_loss.update(batch_loss.detach().item())
			val_acc.update(MC_Accuracy(batch_pred, batch_target))

			desc = f"Epoch: {epoch} \t Loss: {train_loss.avg:.3f} \t Acc: {train_acc.avg:.3f} | Loss: {val_loss.avg:.3f} Acc: {val_acc.avg:.3f}"

			if params.model == 'bcnn':
				desc += f' KL: {kl_div.avg:.2f}\t σ:'

			progress.set_description(desc)

