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


class MC_CrossEntropyLoss(torch.nn.Module):

	def __init__(self, reduction='sum'):

		torch.nn.Module.__init__(self)

		self.reduction = reduction

	def forward(self, pred, target):
		assert pred.dim() == 3, f'Prediction should be of shape [MC, BS, C] but is only of shape {pred.shape}'
		assert target.dim() == 1, f'Targets should be of shape [BS] with each correct label but is only of shape {target.shape}'

		MC, BS, C = pred.shape
		assert target.shape == torch.Size([BS]), f"{target.shape=}"

		MC_target = target.repeat(MC).long()

		loss = F.cross_entropy(pred.flatten(0, 1), MC_target, reduction='mean')
		return loss

def MC_Accuracy(pred, target):

	if pred.dim()==3:
		'''
		Prediction with a Monte Carlo approximation pred.shape=[MC, BS, C]
		'''
		assert target.dim()==1, f"{target.dim()=}"
		assert pred.dim()==3, f"{pred.dim()=}"
		assert target.shape[0]==pred.shape[1], f"{target.shape=}!={pred.shape=}[1]={pred.shape[1]}"

		MC, BS, C = pred.shape

		target = target.repeat(MC)

		train_pred = pred.argmax(dim=-1) # [MC, BS, C] -> [MC, BS]
		train_pred = train_pred.flatten() # [MC, BS] -> [MC * BS]

		assert train_pred.shape==target.shape, f"{train_pred.shape=} != {target.shape=}"

		train_pred_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
		batch_acc = train_pred_correct / target.numel()

		return batch_acc

	elif pred.dim()==2:
		'''
		Deterministic prediction
		'''

		assert target.dim() == 1, f"{target.dim()=}"
		assert pred.dim() == 2, f"{pred.dim()=}"
		assert target.shape[0] == pred.shape[0], f"{target.shape=}!={pred.shape=}[1]={pred.shape[1]}"

		BS, C = pred.shape

		# target = target.repeat(MC)

		train_pred = pred.argmax(dim=-1)  # [MC, BS, C] -> [MC, BS]
		# train_pred = train_pred.flatten()  # [MC, BS] -> [MC * BS]

		assert train_pred.shape == target.shape, f"{train_pred.shape=} != {target.shape=}"

		train_pred_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
		batch_acc = train_pred_correct / target.numel()

		return batch_acc


class MC_NLL(torch.nn.Module):

	def __init__(self, num_samples):

		super().__init__()

		self.num_samples = num_samples # rescales the log-likelihood to the full dataset

	def forward(self, pred, target):
		assert pred.dim() == target.dim() + 1

		mu, std = pred.mean(dim=0), pred.std(dim=0)
		assert mu.shape==std.shape==target.shape
		NLL = -Normal(mu, std).log_prob(target).mean()

		NLL = NLL * self.num_samples

		return NLL




