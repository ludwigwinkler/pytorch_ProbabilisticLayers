import future, sys, os, datetime, argparse
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

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


class CIFAR10DataModule(LightningDataModule):
	name = 'cifar10'
	extra_args = {}

	def __init__(
		self,
		data_dir: str = None,
		val_split: int = 5000,
		num_workers: int = 16,
		batch_size: int = 32,
		seed: int = 42,
		*args,
		**kwargs,
	):
		"""
		.. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/
		    Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png
		    :width: 400
		    :alt: CIFAR-10

		Specs:
		    - 10 classes (1 per class)
		    - Each image is (3 x 32 x 32)

		Standard CIFAR10, train, val, test splits and transforms

		Transforms::

		    mnist_transforms = transform_lib.Compose([
			transform_lib.ToTensor(),
			transforms.Normalize(
			    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
			    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
			)
		    ])

		Example::

		    from pl_bolts.datamodules import CIFAR10DataModule

		    dm = CIFAR10DataModule(PATH)
		    model = LitModel()

		    Trainer().fit(model, dm)

		Or you can set your own transforms

		Example::

		    dm.train_transforms = ...
		    dm.test_transforms = ...
		    dm.val_transforms  = ...

		Args:
		    data_dir: where to save/load the data
		    val_split: how many of the training images to use for the validation split
		    num_workers: how many workers to use for loading data
		    batch_size: number of examples per training/eval step
		"""
		super().__init__(*args, **kwargs)

		self.dims = (3, 32, 32)
		self.DATASET = torchvision.datasets.CIFAR10
		self.val_split = val_split
		self.num_workers = num_workers
		self.batch_size = batch_size
		self.seed = seed
		self.data_dir = data_dir if data_dir is not None else os.getcwd()
		self.num_samples = 60000 - val_split

	@property
	def num_classes(self):
		"""
		Return:
		    10
		"""
		return 10

	def prepare_data(self):
		"""
		Saves CIFAR10 files to data_dir
		"""
		self.DATASET(self.data_dir, train=True, download=True, transform=transform_lib.ToTensor(), **self.extra_args)
		self.DATASET(self.data_dir, train=False, download=True, transform=transform_lib.ToTensor(), **self.extra_args)

	def train_dataloader(self):
		"""
		CIFAR train set removes a subset to use for validation
		"""
		transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms

		dataset = self.DATASET(self.data_dir, train=True, download=False, transform=transforms, **self.extra_args)
		train_length = len(dataset)
		dataset_train, _ = random_split(
			dataset,
			[train_length - self.val_split, self.val_split],
			generator=torch.Generator().manual_seed(self.seed)
		)
		loader = DataLoader(
			dataset,
			batch_size=self.batch_size,
			shuffle=True,
			num_workers=self.num_workers,
			drop_last=True,
			pin_memory=True
		)
		return loader

	def val_dataloader(self):
		"""
		CIFAR10 val set uses a subset of the training set for validation
		"""
		transforms = self.default_transforms() if self.val_transforms is None else self.val_transforms()

		dataset = self.DATASET(self.data_dir, train=False, download=False, transform=transforms, **self.extra_args)
		train_length = len(dataset)
		_, dataset_val = random_split(
			dataset,
			[train_length - self.val_split, self.val_split],
			generator=torch.Generator().manual_seed(self.seed)
		)
		loader = DataLoader(
			dataset_val,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			pin_memory=True,
			drop_last=True
		)
		return loader

	def test_dataloader(self):
		"""
		CIFAR10 test set uses the test split
		"""
		transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms

		dataset = self.DATASET(self.data_dir, train=False, download=False, transform=transforms, **self.extra_args)
		loader = DataLoader(
			dataset,
			batch_size=self.batch_size,
			shuffle=False,
			num_workers=self.num_workers,
			drop_last=True,
			pin_memory=True
		)
		return loader

	def default_transforms(self):
		cf10_transforms = transforms.Compose([
			transforms.RandomCrop(32, padding=4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		return cf10_transforms

	# def train_transforms(self):
	# 	cf10_transforms = transforms.Compose([
	# 		transforms.RandomCrop(32, padding=4),
	# 		transforms.RandomHorizontalFlip(p=.40),
	# 		transforms.ToTensor(),
	# 		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
	# 	return cf10_transforms

	def val_transforms(self):
		cf10_transforms = transforms.Compose([
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
		return cf10_transforms
