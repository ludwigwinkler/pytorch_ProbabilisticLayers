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
import math
import torch
from torch.optim.optimizer import Optimizer
from torch.distributions import Normal

from pytorch_ProbabilisticLayers.src.ProbabilisticLayers import MoG, MoGLinear, BayesLinear


class MoGNatGrad_Adam(Optimizer):
	r"""Implements Adam algorithm.

	It has been proposed in `Adam: A Method for Stochastic Optimization`_.

	Arguments:
	    params (iterable): iterable of parameters to optimize or dicts defining
		parameter groups
	    lr (float, optional): learning rate (default: 1e-3)
	    betas (Tuple[float, float], optional): coefficients used for computing
		running averages of gradient and its square (default: (0.9, 0.999))
	    eps (float, optional): term added to the denominator to improve
		numerical stability (default: 1e-8)
	    weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
	    amsgrad (boolean, optional): whether to use the AMSGrad variant of this
		algorithm from the paper `On the Convergence of Adam and Beyond`_
		(default: False)

	.. _Adam\: A Method for Stochastic Optimization:
	    https://arxiv.org/abs/1412.6980
	.. _On the Convergence of Adam and Beyond:
	    https://openreview.net/forum?id=ryQu7f-RZ
	"""

	def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
		     weight_decay=0, amsgrad=False):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 <= betas[0] < 1.0:
			raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
		if not 0.0 <= betas[1] < 1.0:
			raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
		adam_defaults = dict(lr=lr, betas=betas, eps=eps,
				     weight_decay=weight_decay, amsgrad=amsgrad)
		'''
		We want to apply ADAM to all normal parameters but we want to apply the special MoG update rule to MoG layers.
		Thus we seperate them.
		'''

		mog_param_ids 		= [] # ids for separating mog parameters from adam parameters
		mog_param_names 	= [] # collects the naem in order to check the names for debugging
		self.mog_modules 	= [[]] # list of mog_modules, fist one is empty for adam modules

		adam_params 		= []
		mog_params 		= []
		self._fwd_handles 	= []

		for name, mod in model.named_modules():
			if type(mod)==MoG:
				'''Identify parameters in MoG distribution'''
				mog_param_ids+=[id(x) for x in mod.parameters()]
				'''Create distinct&separate parameter lists for each MoG'''
				mog_params+= [list(mod.parameters())] # [[pi_0, mu_0, logscale_0],[pi_1, mu_1, logscale_1], ... ]
			if type(mod)==MoGLinear:
				'''Apply hook to save weights sampled by MoGLinear during forward pass'''
				handle = mod.register_forward_hook(self._save_sampled_weights)
				self._fwd_handles.append(handle)
				self.mog_modules.append(mod) # for self.state[mog_module]

		'''Creates parameter dictionary for each set of mog params'''
		set_of_mog_params = [{'params': params} for params in mog_params]

		for name, param in model.named_parameters(): # goes over all parameters in model
			if id(param) in mog_param_ids:
				mog_param_names.append(name)
			else:
				''' if id(param) not in mog_param_id, add to Adam optimizable parameter list'''
				adam_params+= [param]

		'''One parameter dict for all parameter optimized by Adam update rule'''
		adam_params = [{'params': adam_params}]

		# print(adam_params)
		# exit('@MoG_Adam init')

		'''When going through parameter groups, check whether parameter set is Adam or MoG parameter group'''
		self.param_group_types = []
		params = []
		if len(adam_params)>0:
			params += adam_params
			self.param_group_types += ['adam']
		if len(set_of_mog_params)>0:
			params += set_of_mog_params
			self.param_group_types += ['mog' for i, _ in enumerate(set_of_mog_params)]

		super(MoGNatGrad_Adam, self).__init__(params, adam_defaults)

	def __setstate__(self, state):
		super(MoGNatGrad_Adam, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('amsgrad', False)

	def _save_sampled_weights(self, mod, i, o):
		if mod.training:
			self.state[mod]['z'] = mod.sampled_w

	def adam_step(self, group):

		for p in group['params']:  # all the stored parameters via id(param)

			if p.grad is None:
				continue
			grad = p.grad.data
			if grad.is_sparse:
				raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
			amsgrad = group['amsgrad']

			state = self.state[p]

			# State initialization
			if len(state) == 0:
				state['step'] = 0
				# Exponential moving average of gradient values
				state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
				# Exponential moving average of squared gradient values
				state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
				if amsgrad:
					# Maintains max of all exp. moving avg. of sq. grad. values
					state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

			exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
			if amsgrad:
				max_exp_avg_sq = state['max_exp_avg_sq']
			beta1, beta2 = group['betas']

			state['step'] += 1
			bias_correction1 = 1 - beta1 ** state['step']
			bias_correction2 = 1 - beta2 ** state['step']

			if group['weight_decay'] != 0:
				grad.add_(p.data, group['weight_decay'])

			# Decay the first and second moment running average coefficient
			exp_avg.mul_(beta1).add_(alpha=1 - beta1, other=grad)
			exp_avg_sq.mul_(beta2).addcmul_(value=1 - beta2, tensor1=grad, tensor2=grad)
			if amsgrad:
				# Maintains the maximum of all 2nd moment running avg. till now
				torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
				# Use the max. for normalizing running avg. of gradient
				denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
			else:
				denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

			step_size = group['lr'] / bias_correction1

			p.data.addcdiv_(value=-step_size, tensor1=exp_avg, tensor2=denom)

	def sgd_step(self, group):

		for p in group['params']:
			# print(f"{p.shape=}")
			if p.grad is None:
				continue
			d_p = p.grad.data

			p.data.add_(-group['lr'], d_p)

	@torch.no_grad()
	def step(self, closure=None):
		"""Performs a single optimization step.

		Arguments:
		    closure (callable, optional): A closure that reevaluates the model
			and returns the loss.
		"""
		loss = None
		'''
		self.param_groups is a list of separate dicts of parameters
		we collect all non MoG parameters in a single set, and every set of MoG parameters in a separate set 
		'''
		for group_i, group in enumerate(self.param_groups):
			param_group_type = self.param_group_types[group_i] # check for the type, e.g. 'adam' or 'mog'
			if param_group_type == 'adam':
				'''Do the standard adam update'''
				# self.adam_step(group)
				self.sgd_step(group)

			elif param_group_type == 'mog':
				''' group.keys()=['params', 'lr'] where params is a list [pi, mu, logscale]'''

				pi, mu, logscale, scales = group['params']
				mog_module = self.mog_modules[group_i]
				sigma = scales.data
				z = self.state[mog_module]['z']
				''' mu.shape/sigma.shape = [num_components, in_features, out_features]'''
				# print(f"{mu.shape=} {logscale.shape=} {z.shape=}")
				p_z = Normal(mu.unsqueeze(1), sigma.unsqueeze(1)).log_prob(z) # -> shape=[num_components, num_MC=1, in, out]

				d_c_nom = p_z
				d_c_denom = torch.sum(p_z*pi.permute(2,0,1).unsqueeze(1), dim=0, keepdim=True)
				d_c = d_c_nom/d_c_denom # shape=[num_components, num_MC, in, out
				d_c = d_c.mean(dim=1)
				# prec = sigma**(-2)
				# assert prec.shape==d_c.shape, f'{prec.shape=} {d_c.shape=}'
				# prec += group['lr']*d_c*mu.grad**2
				# print(f"#"*100)

				# print(f"{mu.data.squeeze()=} {mu.grad.squeeze()=} {F.softplus(logscale.data ).squeeze()=} {F.softplus(logscale.grad).squeeze()=}")
				# print(f"{mu.grad=}")
				# print(f"{z.shape=} {z.mean(dim=0).squeeze()=}")
				# print(f"{scales.data=}")
				# print(f"{group['lr']=}")
				# exit()

				mu.data -= group['lr']*mu.grad #* d_c
				logscale.data -= group['lr']*logscale.grad #* d_c
				# self.adam_step(group)

		# self.zero_grad()



		# exit()

		# for group in self.param_groups:  # different parameters can have different optimization hyperparameters (lr, momentum etc): Type=List
		# 	for p in group['params']:  # all the stored parameters via id(param)
		#
		# 		if p.grad is None:
		# 			continue
		# 		grad = p.grad.data
		# 		if grad.is_sparse:
		# 			raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
		# 		amsgrad = group['amsgrad']
		#
		# 		state = self.state[p]
		#
		# 		# State initialization
		# 		if len(state) == 0:
		# 			state['step'] = 0
		# 			# Exponential moving average of gradient values
		# 			state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
		# 			# Exponential moving average of squared gradient values
		# 			state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
		# 			if amsgrad:
		# 				# Maintains max of all exp. moving avg. of sq. grad. values
		# 				state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
		#
		# 		exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
		# 		if amsgrad:
		# 			max_exp_avg_sq = state['max_exp_avg_sq']
		# 		beta1, beta2 = group['betas']
		#
		# 		state['step'] += 1
		# 		bias_correction1 = 1 - beta1 ** state['step']
		# 		bias_correction2 = 1 - beta2 ** state['step']
		#
		# 		if group['weight_decay'] != 0:
		# 			grad.add_(group['weight_decay'], p.data)
		#
		# 		# Decay the first and second moment running average coefficient
		# 		exp_avg.mul_(beta1).add_(1 - beta1, grad)
		# 		exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
		# 		if amsgrad:
		# 			# Maintains the maximum of all 2nd moment running avg. till now
		# 			torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
		# 			# Use the max. for normalizing running avg. of gradient
		# 			denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
		# 		else:
		# 			denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
		#
		# 		step_size = group['lr'] / bias_correction1
		#
		# 		p.data.addcdiv_(-step_size, exp_avg, denom)

		return loss
