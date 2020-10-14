import future, sys, os, datetime, argparse
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
from torch.nn import Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

def generate_linreg_data(num_samples=1000, x_noise_std=0.01, y_noise_std=0.1, plot=False):

	x = np.linspace(-5, 5, num_samples)
	x += np.random.normal(0., y_noise_std, size=x.shape)
	y_noise = np.random.normal(0., y_noise_std, size=x.shape)

	y = x + y_noise

	x, y = x.reshape(-1, 1), y.reshape(-1, 1)

	if plot:
		plt.scatter(x, y)
		plt.grid()
		plt.show()

	return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def generate_nonstationary_data(num_samples=1000, x_noise_std=0.01, y_noise_std=0.1, plot=False):
	x = np.linspace(-0.35, 0.55, num_samples)
	x_noise = np.random.normal(0., x_noise_std, size=x.shape)

	std = np.linspace(0, y_noise_std, num_samples)  # * y_noise_std
	# print(std.shape)
	non_stationary_noise = np.random.normal(loc=0, scale=std)
	y_noise = non_stationary_noise

	y = x + 0.3 * np.sin(2 * np.pi * (x + x_noise)) + 0.3 * np.sin(4 * np.pi * (x + x_noise)) + y_noise

	x, y = x.reshape(-1, 1), y.reshape(-1, 1)

	if plot:
		plt.scatter(x, y)
		plt.grid()
		plt.show()

	return torch.from_numpy(x).float(), torch.from_numpy(y).float()


def generate_nonstationary_data2(num_samples=1000, x_noise_std=0.01, y_noise_std=0.1, plot=False):
	x = np.linspace(-0.35, 0.55, num_samples)
	x_noise = np.random.normal(0., x_noise_std, size=x.shape)


	std = np.abs(np.cos(10 * np.linspace(0, y_noise_std, num_samples))) * 0.1
	# plt.plot(std)
	# plt.show()
	non_stationary_noise = np.random.normal(loc=0, scale=std)
	y_noise = non_stationary_noise

	y = -0.1 * x ** 2 + y_noise

	x, y = x.reshape(-1, 1), y.reshape(-1, 1)

	if plot:
		plt.scatter(x, y)
		plt.show()
		exit()

	return torch.from_numpy(x).float(), torch.from_numpy(y).float()