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
from torch.distributions import Normal
from torch.nn import Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD


def MC_CrossEntropy(pred, target):
	assert pred.dim() == 3, f'Prediction should be of shape [MC, BS, C] but is only of shape {pred.shape}'
	assert target.dim() == 1, f'Targets should be of shape [BS] with each correct label but is only of shape {target.shape}'

	num_MC, batch_size, num_C = pred.shape

	MC_target = FloatTensor(batch_size, num_C).fill_(0)
	MC_target.scatter_(1, target.unsqueeze(-1), 1)
	MC_target = MC_target.unsqueeze(0).repeat((num_MC, 1, 1))

	loss = F.cross_entropy(pred.flatten(0, 1), target.unsqueeze(0).repeat(num_MC, 1).flatten(0, 1).long())
	return loss

class MC_NLL(torch.nn.Module):

	def __init__(self, reduction='sum'):

		super().__init__()

		self.reduction = reduction

	def forward(self, pred, target):
		assert pred.dim() == target.dim() + 1

		mu, std = pred.mean(dim=0), pred.std(dim=0)
		assert mu.shape==std.shape==target.shape
		NLL = -Normal(mu, std).log_prob(target)

		if self.reduction == 'sum':
			NLL = NLL.sum()
		elif self.reduction == 'mean':
			NLL = NLL.mean()

		return NLL




