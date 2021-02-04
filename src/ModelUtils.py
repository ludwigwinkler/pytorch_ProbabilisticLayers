import future, sys, os, datetime, argparse
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from math import prod
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch, torchvision
import pytorch_lightning as light
from torch.nn import Module, Parameter, Sequential
from torch.nn import Tanh, ReLU, Sigmoid, LeakyReLU
from torch.nn import Linear, Conv2d, MaxPool2d, Dropout, BatchNorm2d
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, Independent

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=5, sci_mode=False)
np.set_printoptions(precision=5, suppress=True)

sys.path.append("../..")  # Up to -> KFAC -> Optimization -> PHD

from argparse import ArgumentParser

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

import os

from Utils.Utils import str2bool

import torch
from torch.nn import functional as F, Sequential, Linear
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as light
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.functional import accuracy

class PrintModule(torch.nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, x):
		print(f"PrintModule: {x.shape=}")
		return x


class AdaptiveInit_FlattenAndLinear(torch.nn.Module):

	def __init__(self, num_hidden):
		super(AdaptiveInit_FlattenAndLinear, self).__init__()
		self.adaptively_initialized = False
		self.num_hidden = num_hidden

	def forward(self, x):

		if not self.adaptively_initialized:
			self.adaptively_initialized = True

			x_dim = x.shape
			x = x.flatten(1, -1)

			self.linear = Linear(x.shape[-1], self.num_hidden)
			out = self.linear(x)
			# print(f"FlattenAndLinear: {list(x_dim[1:])} ({prod(x_dim[1:])}) -> {list(out.shape[1:])}")

		else:
			x = x.flatten(1,-1)
			out = self.linear(x)
		return out

class CIFAR10_ReferenceNet(Module):
	def __init__(self):
		super().__init__()
		self.conv1 	= Conv2d(3, 6, 5)
		self.pool 	= MaxPool2d(2, 2)
		self.conv2 	= Conv2d(6, 16, 5)
		self.fc1 	= Linear(16 * 5 * 5, 120)
		self.fc2 	= Linear(120, 84)
		self.fc3 	= Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class CIFAR10_ResNet18(Module):
	def __init__(self):
		super().__init__()
		base = torchvision.models.resnet18(pretrained=True)
		self.base = Sequential(*list(base.children())[:-1])
		in_features = base.fc.in_features
		self.drop = Dropout()
		self.final = Linear(in_features, 10)

	def forward(self, x):
		x = self.base(x)
		x = torch.flatten(x, -3)
		return self.final(x)

def ResNet18(in_C):
	return ResNet(in_C, BasicBlock, [2, 2, 2, 2])

class BasicBlock(torch.nn.Module):
	expansion = 1

	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = Conv2d(
			in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = BatchNorm2d(planes)
		self.conv2 = Conv2d(planes, planes, kernel_size=3,
				       stride=1, padding=1, bias=False)
		self.bn2 = BatchNorm2d(planes)

		self.shortcut = Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = Sequential(
				Conv2d(in_planes, self.expansion * planes,
					  kernel_size=1, stride=stride, bias=False),
				BatchNorm2d(self.expansion * planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class ResNet(torch.nn.Module):
	def __init__(self, in_C, block, num_blocks, num_classes=10):
		super(ResNet, self).__init__()
		self.in_planes = 64

		self.conv1 = torch.nn.Conv2d(in_C, 64, kernel_size=3,
				       stride=1, padding=1, bias=False)
		self.bn1 = torch.nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.linear = Linear(512 * block.expansion, num_classes)

	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return Sequential(*layers)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		out = out.view(out.size(0), -1)
		out = self.linear(out)
		return out