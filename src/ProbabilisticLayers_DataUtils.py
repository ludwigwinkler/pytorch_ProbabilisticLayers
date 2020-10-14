import future, sys, os, datetime, argparse
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

from backpack import extend

import torch, torchvision
import pytorch_lightning as light
from torch.nn import Module, Parameter
from torch.nn import Linear, Tanh, ReLU, Sigmoid, LeakyReLU
import torch.nn.functional as F
from torch.distributions import Bernoulli, Normal, Independent

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=10, sci_mode=False)
np.set_printoptions(precision=10, suppress=True)

sys.path.append("../..")  # Up to -> KFAC -> Optimization -> PHD

from argparse import ArgumentParser

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

import os

from Utils.Utils import str2bool

import torch
from torch.nn import functional as F, Sequential, Linear
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, FashionMNIST
from torchvision import transforms

import pytorch_lightning as light
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.metrics.functional import accuracy


class MNISTDataModule(LightningDataModule):

	def __init__(self, hparams):
		super().__init__()

		self.data_str = '../data/MNIST'
		self.hparams = hparams

		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.1307,), (0.3081,))
		])

		self.num_classes = 10
		self.in_features = 28 * 28

		self.dims = (1, 28, 28)
		self.num_samples = 55000

	def prepare_data(self, *args, **kwargs):
		MNIST(self.data_str, train=True, download=True)
		MNIST(self.data_str, train=False, download=True)

	def setup(self, stage=None):
		if stage == 'fit' or stage is None:
			mnist_full = MNIST(self.data_str, train=True, transform=self.transform)
			self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
			self.num_samples = 55000

	def train_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True, drop_last=True)

	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size, drop_last=True, num_workers=self.hparams.num_workers)

class FMNISTDataModule(LightningDataModule):

	def __init__(self, hparams):
		super().__init__()

		self.data_str = '../data/FMNIST'
		self.hparams = hparams

		self.transform = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.5,), (0.5,))
		])

		self.num_classes = 10
		self.in_features = 28 * 28

		self.dims = (1, 28, 28)
		self.num_samples = 55000

	def prepare_data(self, *args, **kwargs):
		FashionMNIST(self.data_str, train=True, download=True)
		FashionMNIST(self.data_str, train=False, download=True)

	def setup(self, stage=None):
		if stage == 'fit' or stage is None:
			mnist_full = FashionMNIST(self.data_str, train=True, transform=self.transform)
			self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
			self.num_samples = 55000

	def train_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.mnist_train, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, shuffle=True, pin_memory=True, drop_last=True)

	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size, drop_last=True, num_workers=self.hparams.num_workers)
