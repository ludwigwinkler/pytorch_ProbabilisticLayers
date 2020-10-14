import torch
from torch.distributions import Normal, Dirichlet, Categorical, RelaxedOneHotCategorical
from torch.nn import Sequential, Tanh, ReLU, Linear, Dropout, CELU, BatchNorm1d, Parameter
import torch.nn.functional as F

import numpy as np
from numbers import Number
import math
import matplotlib.pyplot as plt

# torch.manual_seed(1234)

if torch.cuda.is_available():
	FloatTensor = torch.cuda.FloatTensor
	Tensor = torch.cuda.FloatTensor
elif not torch.cuda.is_available():
	FloatTensor = torch.FloatTensor
	Tensor = torch.FloatTensor

#Loss Functions
class MC_CrossEntropyLoss(torch.nn.Module):

	def __init__(self, num_samples):
		torch.nn.Module.__init__(self)

		assert num_samples > 0
		assert type(num_samples) is int

		self.num_samples = num_samples

	def forward(self, pred, target):
		assert pred.dim() == 3, f'Prediction should be of shape [MC, BS, C] but is only of shape {pred.shape}'
		assert target.dim() == 1, f'Targets should be of shape [BS] with each correct label but is only of shape {target.shape}'

		MC, BS, C = pred.shape
		assert target.shape == torch.Size([BS]), f"{target.shape=}"

		MC_target = target.repeat(MC).long()

		loss = F.cross_entropy(pred.flatten(0, 1), MC_target, reduction='mean') * self.num_samples

		return loss


def MC_Accuracy(pred, target):
	if pred.dim() == 3:
		'''
		Prediction with a Monte Carlo approximation pred.shape=[MC, BS, C]
		'''
		assert target.dim() == 1, f"{target.dim()=}"
		assert pred.dim() == 3, f"{pred.dim()=}"
		assert target.shape[0] == pred.shape[1], f"{target.shape=}!={pred.shape=}[1]={pred.shape[1]}"

		MC, BS, C = pred.shape

		target = target.repeat(MC)

		train_pred = pred.argmax(dim=-1)  # [MC, BS, C] -> [MC, BS]
		train_pred = train_pred.flatten()  # [MC, BS] -> [MC * BS]

		assert train_pred.shape == target.shape, f"{train_pred.shape=} != {target.shape=}"

		train_pred_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
		batch_acc = train_pred_correct / target.numel()

		return Tensor([batch_acc])

	elif pred.dim() == 2:
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

		return Tensor([batch_acc])


class MC_NLL(torch.nn.Module):

	def __init__(self, num_samples):
		super().__init__()

		self.num_samples = num_samples  # rescales the log-likelihood to the full dataset

	def forward(self, pred, target):
		assert pred.dim() == target.dim() + 1

		mu, std = pred.mean(dim=0), pred.std(dim=0) + 1e-3
		# std = torch.ones_like(std).fill_(0.1)
		assert mu.shape == std.shape == target.shape, f'{mu.shape=} {std.shape=} {target.shape=}'
		'''
		We compute the average PER-SAMPLE loss ...
		'''
		NLL = -Normal(mu, std).log_prob(target).mean()
		# print(f'{mu=} {std=} -> {NLL=}')
		# exit()
		'''
		... and scale it up to the total number of samples
		'''
		NLL = NLL * self.num_samples

		return NLL

#Prior

class GaussianPrior:

	def __init__(self, scale):
		assert scale>0.

		self.scale = scale

	def dist(self):
		return Normal(0, self.scale)

class LaplacePrior(torch.nn.Module):
	'''
	Attach a tensor.register_hook() call to the relevant parameter
	Store gradient and compute Laplace Approximation
	Compute KL-Divergence
	'''
	def __init__(self, module, clamp=False):

		super().__init__()

		# self.module = module
		self.scale = torch.ones_like(module.weight.loc.data)
		module.weight.loc.register_hook(self._save_grad)

		self.clamp = clamp
		self.step = 0
		self.beta = 0.99

	def _save_grad(self, grad):

		self.step += 1
		bias_correction = 1 - self.beta ** self.step
		self.scale.mul_(self.beta).add_(alpha=1-self.beta, other=(1/grad.data**2).div_(bias_correction+1e-8))

	def dist(self):
		if self.clamp:
			return Normal(0, torch.clamp(self.scale**0.5, min=1.0))
		else:
			return Normal(0, self.scale ** 0.5+1e-8)

#Activation functions
class ShiftedReLU(torch.nn.Module):

	def __init__(self, offset=-1):
		super().__init__()

		self.offset = offset

	def forward(self, x):
		out = torch.clamp(input=x, min=self.offset)

		return out

	def plot_activation(self):
		x = torch.linspace(-10, 10, 1000)

		y = self.forward(x)

		plt.plot(x.numpy(), y.numpy())
		plt.grid()
		plt.show()

class ShiftedLeakyReLU(torch.nn.Module):

	def __init__(self, _offset=-1, _slope=0.1):
		super().__init__()

		self.offset = _offset
		self.slope = _slope

	def forward(self, x):
		out = torch.clamp(input=x, min=self.offset) + self.slope * x * x.le(
			self.offset).float() - self.slope * self.offset * x.le(self.offset).float()

		return out

	def plot_activation(self):
		x = torch.linspace(-10, 10, 1000)

		y = self.forward(x)

		plt.plot(x.numpy(), y.numpy())
		plt.grid()
		plt.show()

#Other Layers
class MC_MaxPool2d(torch.nn.Module):

	def __init__(self, kernel_size=2, num_MC=None):

		super().__init__()

		self.kernel_size=kernel_size

	def forward(self, x, num_MC=None):

		assert x.dim() == 5, 'Input tensor does not have dimensionality of 5 [num_MC, Batch_Size, Ci, H, W]'
		if num_MC != None:
			assert isinstance(num_MC, int)
			assert x.shape[0]==num_MC

		num_MC              = x.shape[0]
		batch_size          = x.shape[1]
		channels            = x.shape[2]

		out = x.permute(1,0,2,3,4) # shape = [Batch_Size, num_MC, dim_in, height, width]
		out = out.flatten(1,2).contiguous() # shape = [batch_size, num_MC * dim_in, height, width]
		out = F.max_pool2d(out, self.kernel_size, self.kernel_size)
		out = out.reshape(batch_size, num_MC, channels, out.shape[-2], out.shape[-1]).contiguous()
		out = out.permute(1,0,2,3,4).contiguous()
		# out = out.chunk(num_MC,1) # num_MC tuple of [ batch_size, dim_out, height, width]
		# out = torch.stack(out,dim=0) # shape = [num_MC, batch_size, dim_out, height, width]


		return out

def MC_dropout1d(x, _p, _num_MC=None):

	assert x.dim() == 4, 'Input tensor does not have dimensionality of 5 [num_MC, Batch_Size, Ci, H, W]'
	if _num_MC != None: assert x.shape[0]==_num_MC

	num_MC = x.shape[0]
	batch_size = x.shape[1]
	dim_out = x.shape[2]

def MC_dropout2d(x, _p, _num_MC=None):

	assert x.dim() == 5, 'Input tensor does not have dimensionality of 5 [num_MC, Batch_Size, Ci, H, W]'
	if _num_MC != None: assert x.shape[0]==_num_MC

	num_MC = x.shape[0]
	batch_size = x.shape[1]
	dim_out = x.shape[2]

# SIVI Layers:
class SIVILayer(torch.nn.Module):

	def init_weights(self, _module):
		if type(_module) == torch.nn.Linear:
			torch.nn.init.orthogonal_(_module.weight, gain=1.)
			if type(_module.bias) != None:
				_module.bias.data.normal_(0., 0.01)

	def __init__(self, in_features, out_features, _dim_noise_input):

		super().__init__()

		self.dim_input = in_features
		self.dim_output = out_features
		self.dim_noise_input = _dim_noise_input

		self.dim_output_params = in_features * out_features * 2 + out_features * 2
		self.num_hidden = np.min([self.dim_output_params, int((_dim_noise_input + self.dim_output_params) / 2)])

		self.prior_sigma = torch.scalar_tensor(1.0)

		self.sivi_net = Sequential(Linear(self.dim_noise_input, self.num_hidden),
		                           Tanh(),
		                           Linear(self.num_hidden, self.num_hidden),
		                           Tanh(),
		                           Linear(self.num_hidden, self.dim_output_params))  # weight matrix x mu x logsigma + bias x mu x logsigma

		self.noise_dist = Normal(loc=torch.zeros((self.dim_noise_input,)), scale=torch.ones((self.dim_noise_input,)))
		self.sivi_net.apply(self.init_weights)

	def forward(self, x):
		'''

		:param x: 3 dimensional tensor of shape [N_MC, M_BatchSize, d_Input]
		:return:
		'''

		assert x.dim() == 3, 'Input tensor not of shape [N_MC, BatchSize, Features]'
		# assert x.shape[0]==_base_noise.shape[0], 'Input and base_noise should have the same number of num_MC samples'
		# assert _base_noise.shape[1]==self.dim_noise_input

		num_MC = x.shape[0]
		batch_size = x.shape[1]

		noise = self.noise_dist.sample((num_MC,)).to(next(self.sivi_net.parameters()).device)
		sivi_out = self.sivi_net(noise)
		sivi_w = sivi_out[:, :self.dim_input * self.dim_output * 2]
		sivi_b = sivi_out[:, self.dim_input * self.dim_output * 2:]

		w_mu, w_logsigma = torch.chunk(sivi_w, chunks=2, dim=-1)
		self.w_mu = w_mu.reshape((num_MC, self.dim_input, self.dim_output))
		self.w_std = F.softplus(w_logsigma.reshape((num_MC, self.dim_input, self.dim_output)))

		self.b_mu, b_logsigma = torch.chunk(sivi_b, chunks=2, dim=-1)
		self.b_std = F.softplus(b_logsigma)

		dist_w = Normal(self.w_mu, self.w_std)
		dist_b = Normal(self.b_mu, self.b_std)

		# print(dist_w)

		self.sampled_w = dist_w.rsample()
		self.sampled_b = dist_b.rsample()

		# print(f"{x.shape=}")
		# print(f"{self.sampled_w.shape=}")
		# print(f"{self.sampled_b.shape=}")

		# out = torch.bmm(x, dist_w.rsample()) + dist_b.rsample().unsqueeze(1)
		out = torch.baddbmm(self.sampled_b.unsqueeze(-1), x, self.sampled_w)
		# out = torch.bmm(self.sampled_b, x, self.sampled_w)

		# exit()
		prior_w = torch.distributions.kl_divergence(dist_w, Normal(0,1.))
		prior_b = torch.distributions.kl_divergence(dist_b, Normal(0,1.))

		self.kl_div = prior_w.mean(dim=0).sum() #+ prior_b.mean(dim=0).sum()

		return out

	def sample_posterior_dist(self, _samples=2000, _plot=True, _str=''):

		with torch.no_grad():

			sivi_noise = self.noise_dist.sample(sample_shape=(_samples,)).float()
			sivi_out = self.sivi_net(sivi_noise)
			sivi_w = sivi_out[:, :self.dim_input * self.dim_output * 2]
			sivi_b = sivi_out[:, self.dim_input * self.dim_output * 2:]

			w_mu, w_logsigma = torch.chunk(sivi_w, chunks=2, dim=-1)
			self.w_mu = w_mu.reshape((_samples, self.dim_input, self.dim_output))
			self.w_std = F.softplus(w_logsigma.reshape((_samples, self.dim_input, self.dim_output)))

			self.b_mu, b_logsigma = torch.chunk(sivi_b, chunks=2, dim=-1)
			self.b_std = F.softplus(b_logsigma)

			dist_w = Normal(self.w_mu, self.w_std)
			dist_b = Normal(self.b_mu, self.b_std)

			w = dist_w.sample()
			b = dist_b.sample()

			if _plot:
				w = w.flatten(start_dim=1, end_dim=2).T

				if w.shape[0] > 3:
					ncols_nrows = int(w.shape[0] ** 0.5)
					# odd_offset = w.shape[0]%2
					fig, axs = plt.subplots(nrows=ncols_nrows, ncols=ncols_nrows, figsize=(10, 10), sharex=True, sharey=True)
				else:
					fig, axs = plt.subplots(nrows=w.shape[0], ncols=1, figsize=(10, 10), sharex=True, sharey=True)
				fig.suptitle(_str)

				axs = axs.flat

				for i in range(w.shape[0]-1):
					axs[i].hist(w[i].numpy(), bins=150, density=True, color='r', alpha=0.5)
					axs[i].set_ylim(0, 5)

				plt.show()

class JointSIVILayer(torch.nn.Module):

	def init_weights(self, _module):
		if type(_module) == torch.nn.Linear:
			torch.nn.init.orthogonal_(_module.weight, gain=1.)
			if type(_module.bias) != None:
				_module.bias.data.normal_(0., 0.01)

	def __init__(self,
	             _dim_in=-1,
	             _dim_out=-1,
	             _dim_noise_in=-1,
	             _prior_sigma=-1,
	             _single_logstd=True):

		super().__init__()

		self.dim_in = _dim_in
		self.dim_out = _dim_out
		self.dim_noise_input = _dim_noise_in

		self.single_logstd = _single_logstd

		self.prior_sigma = _prior_sigma

		if _single_logstd:
			self.dim_output_params = _dim_in * _dim_out + _dim_out
			self.w_logsigma = Parameter(torch.scalar_tensor(-3.))
			self.b_logsigma = Parameter(torch.scalar_tensor(-3.))
		elif not _single_logstd:
			self.dim_output_params = _dim_in * _dim_out * 2 + _dim_out * 2
		self.num_hidden = np.min([self.dim_output_params, int((_dim_noise_in + self.dim_output_params) / 2)])

		self.sivi_net = Sequential(Linear(self.dim_noise_input, self.num_hidden),
		                           Tanh(),
		                           Linear(self.num_hidden, self.num_hidden),
		                           Tanh(),
		                           Linear(self.num_hidden, self.dim_output_params))  # weight matrix x mu x logsigma + bias x mu x logsigma
		print('JointSIVIFC')
		print(self.sivi_net)
	# self.sivi_net.apply(self.init_weights)

	def forward(self, x: torch.Tensor, _prior, _base_noise):
		'''

		:param x: 3 dimensional tensor of shape [N_MC, M_BatchSize, d_Input]
		:return:
		'''

		assert x.dim() == 3, 'Input tensor not of shape [N_MC, BatchSize, Features]'
		assert x.shape[0] == _base_noise.shape[0], 'Input and base_noise should have the same number of num_MC samples'
		assert _base_noise.shape[1] == self.dim_noise_input

		num_MC = x.shape[0]
		batch_size = x.shape[1]

		sivi_out = self.sivi_net(_base_noise)
		# print('sivi_out', sivi_out.shape)
		if self.single_logstd:
			w_mu, b_mu = sivi_out.split(self.dim_in * self.dim_out,dim=1) # splits sivi_generator output into two pieces
			w_mu = w_mu.reshape((num_MC,self.dim_in, self.dim_out))
			b_mu = b_mu.reshape((num_MC,self.dim_out))
			w_std = F.softplus(self.w_logsigma).expand_as(w_mu)
			b_std = F.softplus(self.b_logsigma).expand_as(b_mu)
		elif not self.single_logstd:
			w, b = sivi_out.split(self.dim_in * self.dim_out * self.kernel_size**2 * 2, dim=1)

			w_mu, w_logsigma = torch.chunk(w, chunks=2, dim=-1)
			b_mu, b_logsigma = torch.chunk(b, chunks=2, dim=-1)

			w_mu = w_mu.reshape((num_MC*self.dim_out, self.dim_in, self.kernel_size, self.kernel_size))
			w_std = F.softplus(w_logsigma.reshape((num_MC*self.dim_out, self.dim_in, self.kernel_size, self.kernel_size)))

			b_mu = b_mu.reshape((num_MC*self.dim_out))
			b_std = F.softplus(b_logsigma).reshape((num_MC*self.dim_out))

		# print(w_mu.shape, w_std.shape)
		# print(b_mu.shape, b_std.shape)

		dist_w = Normal(w_mu, w_std)
		dist_b = Normal(b_mu, b_std)

		w = dist_w.rsample()
		b = dist_b.rsample()

		out = torch.bmm(x, w) + b.unsqueeze(1)
		# exit()

		prior_w = torch.distributions.kl_divergence(dist_w, Normal(torch.zeros_like(dist_w.loc), self.prior_sigma * torch.ones_like(dist_w.scale)))
		# prior_b = torch.distributions.kl_divergence(dist_b, Normal(torch.zeros_like(self.b_mu), self.prior_sigma*torch.ones_like(self.b_std)))
		prior = _prior + (prior_w.mean(dim=0).sum())

		return out, None, prior.squeeze()

	def sample_posterior_dist(self, _num_samples=500, _plot=True, _str='', _base_sivi_noise=None):

		assert _base_sivi_noise.dim() == 2, 'Base noise tensor not of shape [N_MC, dim_noise]'

		_num_samples=_base_sivi_noise.shape[0]

		with torch.no_grad():

			sivi_out = self.sivi_net(_base_sivi_noise)
			if self.single_logstd:
				w_mu, b_mu = sivi_out.split(self.dim_in * self.dim_out,dim=1) # splits sivi_generator output into two pieces
				w_mu = w_mu.reshape((_num_samples,self.dim_in, self.dim_out))
				b_mu = b_mu.reshape((_num_samples,self.dim_out))
				w_std = F.softplus(self.w_logsigma).expand_as(w_mu)
				b_std = F.softplus(self.b_logsigma).expand_as(b_mu)
			elif not self.single_logstd:
				w, b = sivi_out.split(self.dim_in * self.dim_out * self.kernel_size**2 * 2, dim=1)

				w_mu, w_logsigma = torch.chunk(w, chunks=2, dim=-1)
				b_mu, b_logsigma = torch.chunk(b, chunks=2, dim=-1)

				w_mu = w_mu.reshape((_num_samples*self.dim_out, self.dim_in, self.kernel_size, self.kernel_size))
				w_std = F.softplus(w_logsigma.reshape((_num_samples*self.dim_out, self.dim_in, self.kernel_size, self.kernel_size)))

				b_mu = b_mu.reshape((_num_samples*self.dim_out))
				b_std = F.softplus(b_logsigma).reshape((_num_samples*self.dim_out))

			dist_w = Normal(w_mu, w_std)

			w = dist_w.sample()

			if _plot:
				w = w.flatten(start_dim=1, end_dim=2).T

				if w.shape[0] > 3:
					ncols_nrows = int(w.shape[0] ** 0.5) + 1
					# odd_offset = w.shape[0]%2
					fig, axs = plt.subplots(nrows=ncols_nrows, ncols=ncols_nrows, figsize=(10, 10), sharex=True, sharey=True)
				else:
					fig, axs = plt.subplots(nrows=w.shape[0], ncols=1, figsize=(10, 10), sharex=True, sharey=True)
				fig.suptitle(_str)

				axs = axs.flat

				for i in range(w.shape[0]):
					axs[i].hist(w[i].numpy(), bins=150, density=True, color='r', alpha=0.5)
				# axs[i].set_ylim(0,2)

				plt.show()
		return fig

class ConvSIVILayer(torch.nn.Module):

	def __init__(self, _dim_in=-1, _dim_out=-1, _dim_noise_input=10, _kernel_size=1, _stride=1):
		super().__init__()

		self.dim_in         = _dim_in
		self.dim_out        = _dim_out
		self.dim_noise_in   = _dim_noise_input
		self.kernel_size    = _kernel_size
		self.stride         = _stride

		self.dim_params = _dim_in * _dim_out * _kernel_size * _kernel_size * 2 + _dim_out * 2
		self.num_hidden = np.min([self.dim_params, int((_dim_noise_input + self.dim_params) / 2)])
		self.prior_sigma = torch.scalar_tensor(1.0)
		self.sivi_net = Sequential(Linear(self.dim_noise_in, self.num_hidden),
		                           ReLU(),
		                           Linear(self.num_hidden, self.num_hidden),
		                           ReLU(),
		                           Linear(self.num_hidden, self.dim_params))  # weight matrix x mu x logsigma + bias x mu x logsigma

		self.noise_dist = Normal(loc=torch.zeros((self.dim_noise_in,)), scale=torch.ones((self.dim_noise_in,)))
	# self.sivi_net.apply(self.init_weights)

	def forward(self, x: torch.Tensor, _prior):

		assert x.dim() == 5, 'Input tensor not of shape [Num_MC, BatchSize, Features, height, width]'

		num_MC = x.shape[0]
		batch_size = x.shape[1]
		out = x.permute(1,0,2,3,4)
		out = out.flatten(1,2).contiguous()

		noise = self.noise_dist.sample((num_MC,)).to(next(self.sivi_net.parameters()).device)
		sivi_out = self.sivi_net(noise)

		w, b = sivi_out.split(self.dim_in * self.dim_out * self.kernel_size**2 * 2, dim=1)

		w_mu, w_logsigma = torch.chunk(w, chunks=2, dim=-1)
		b_mu, b_logsigma = torch.chunk(b, chunks=2, dim=-1)

		w_mu = w_mu.reshape((num_MC*self.dim_out, self.dim_in, self.kernel_size, self.kernel_size))
		w_std = F.softplus(w_logsigma.reshape((num_MC*self.dim_out, self.dim_in, self.kernel_size, self.kernel_size)))

		b_mu = b_mu.flatten()
		b_std = F.softplus(b_logsigma).flatten()
		# print(b_mu.shape, b_logsigma.shape)
		# exit()

		dist_w = Normal(w_mu, w_std)
		dist_b = Normal(b_mu, b_std)

		w = dist_w.rsample()
		b = dist_b.rsample()

		out = F.conv2d(input=out, weight=w, bias=b, groups=num_MC, stride=1) # shape=[batch_size, num_MC*dim_out, height, width]
		print(out.shape)

		out = out.reshape(batch_size, num_MC, self.dim_out, out.shape[-2], out.shape[-1])
		out = out.permute(1,0,2,3,4)
		# out = out.chunk(num_MC,1) # num_MC tuple of [ batch_size, dim_out, height, width]
		# out = torch.stack(out,dim=0) # shape = [num_MC, batch_size, dim_out, height, width]
		# print(out.shape)

		# exit()

		prior = _prior + torch.distributions.kl_divergence(Normal(torch.zeros_like(w_mu), self.prior_sigma*torch.ones_like(w_mu)), dist_w).mean(dim=0).sum()

		return out, None, prior

class JointSIVICNNLayer(torch.nn.Module):

	def __init__(self,
	             _dim_in=-1,
	             _dim_out=-1,
	             _dim_noise_input=10,
	             _dim_noise_hidden_layers=0,
	             _kernel_size=1,
	             _stride=1,
	             _prior_sigma=1.0,
	             _single_logstd=True):

		super().__init__()

		self.dim_in         = _dim_in
		self.dim_out        = _dim_out
		self.dim_noise_in   = _dim_noise_input
		self.kernel_size    = _kernel_size
		self.stride         = _stride

		self.single_logstd = _single_logstd

		self.prior_sigma = _prior_sigma

		if _single_logstd:
			self.dim_params = _dim_in * _dim_out * _kernel_size * _kernel_size + _dim_out
			self.w_logsigma = Parameter(torch.scalar_tensor(-3.))
			self.b_logsigma = Parameter(torch.scalar_tensor(-3.))
		elif not _single_logstd:
			self.dim_params = _dim_in * _dim_out * _kernel_size * _kernel_size * 2 + _dim_out * 2
		self.num_hidden = np.min([self.dim_params, int((_dim_noise_input + self.dim_params) / 2)])
		self.prior_sigma = torch.scalar_tensor(1.0)

		sivi_net = []
		sivi_net.extend([Linear(_dim_noise_input, self.num_hidden),Tanh()])
		[sivi_net.extend([Linear(self.num_hidden, self.num_hidden),Tanh()]) for _ in range(_dim_noise_hidden_layers)]
		sivi_net.extend([Linear(self.num_hidden, self.dim_params)])
		self.sivi_net = torch.nn.Sequential(*sivi_net)

		# self.sivi_net = Sequential(Linear(self.dim_noise_in, self.num_hidden),
		#                            Tanh(),
		# Linear(self.num_hidden, self.num_hidden),
		# Tanh(),
		# Linear(self.num_hidden, self.dim_params))  # weight matrix x mu x logsigma + bias x mu x logsigma
		print('JointSIVICNN')
		print(self.sivi_net)
		self.noise_dist = Normal(loc=torch.zeros((self.dim_noise_in,)), scale=torch.ones((self.dim_noise_in,)))
	# self.sivi_net.apply(self.init_weights)

	def forward(self, x: torch.Tensor, _prior, _base_noise):

		assert x.dim() == 5, 'Input tensor not of shape [Num_MC, BatchSize, Features, height, width]'
		assert _base_noise.shape[0] == x.shape[0], 'Base noise num_MC is different from x num_MC'
		assert _base_noise.shape[1] == self.dim_noise_in, 'Base noise dim is different from predefined noise dim'


		num_MC = x.shape[0]
		batch_size = x.shape[1]
		out = x.permute(1,0,2,3,4)
		out = out.flatten(1,2).contiguous()

		sivi_out = self.sivi_net(_base_noise)
		# print(sivi_out.shape)

		if self.single_logstd:
			w_mu, b_mu = sivi_out.split(self.dim_in * self.dim_out * self.kernel_size**2, dim=1) # splits sivi_generator output into two pieces
			w_mu = w_mu.reshape((num_MC*self.dim_out, self.dim_in, self.kernel_size, self.kernel_size))
			b_mu = b_mu.reshape((num_MC*self.dim_out))
			w_std = F.softplus(self.w_logsigma).expand_as(w_mu)
			b_std = F.softplus(self.b_logsigma).expand_as(b_mu)
		elif not self.single_logstd:
			w, b = sivi_out.split(self.dim_in * self.dim_out * self.kernel_size**2 * 2, dim=1)

			w_mu, w_logsigma = torch.chunk(w, chunks=2, dim=-1)
			b_mu, b_logsigma = torch.chunk(b, chunks=2, dim=-1)

			w_mu = w_mu.reshape((num_MC*self.dim_out, self.dim_in, self.kernel_size, self.kernel_size))
			w_std = F.softplus(w_logsigma.reshape((num_MC*self.dim_out, self.dim_in, self.kernel_size, self.kernel_size)))

			b_mu = b_mu.reshape((num_MC*self.dim_out))
			b_std = F.softplus(b_logsigma).reshape((num_MC*self.dim_out))

		# print(w_mu.shape, w_std.shape)
		# print(b_mu.shape, b_std.shape)
		# exit()
		dist_w = Normal(w_mu, w_std)
		dist_b = Normal(b_mu, b_std)

		w = dist_w.rsample()
		b = dist_b.rsample()

		# print(w.shape, b.shape)

		out = F.conv2d(input=out, weight=w, groups=num_MC, bias=b, stride=self.stride)
		# print('1', out.shape)

		out = out.reshape(batch_size, num_MC, self.dim_out, out.shape[-2], out.shape[-1])
		out = out.permute(1,0,2,3,4)
		# out = out.chunk(num_MC,1) # num_MC tuple of [ batch_size, dim_out, height, width]
		# out = torch.stack(out,dim=0) # shape = [num_MC, batch_size, dim_out, height, width]
		# print(out.shape)

		prior = _prior + torch.distributions.kl_divergence(Normal(torch.zeros_like(w_mu), self.prior_sigma*torch.ones_like(w_mu)), dist_w).mean(dim=0).sum()

		return out, None, prior

#BatchNorm Layers
class MC_BatchNorm1D(torch.nn.modules.batchnorm._BatchNorm):

	def forward(self, x):

		assert x.dim()==3, 'Input tensor does not have dimensionality of 5 [num_MC, Batch_Size, Features]'
		assert self.num_features==x.shape[-1]

		MC = x.shape[0]
		BS = x.shape[1]
		Features = x.shape[2]

		# out = x.permute(1,2,0) # shape = [batch_size, features, MC] since batchnorm computes statistics for second dimension
		out = x.flatten(0,1).contiguous() # shape = [MC*batch_size, features]
		# print(out)
		if self.momentum is None:
			exponential_average_factor = 0.0
		else:
			exponential_average_factor = self.momentum

		if self.training and self.track_running_stats:
			# TODO: if statement only here to tell the jit to skip emitting this when it is None
			if self.num_batches_tracked is not None:
				self.num_batches_tracked = self.num_batches_tracked + 1
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		out = F.batch_norm( out, self.running_mean, self.running_var, self.weight, self.bias,
		# out = F.batch_norm( out, self.running_mean, self.running_var, torch.Tensor([1.]), torch.Tensor([0.]),
				self.training or not self.track_running_stats,
				exponential_average_factor, self.eps)
		# out = out.reshape()
		out = out.reshape(MC, BS, Features).contiguous()
		# out = out.permute(2,0,1) # shape = [MC, batch_size, features]
		return out

class MC_BatchNorm2D(torch.nn.modules.batchnorm._BatchNorm):

	def forward(self, x):

		assert x.dim()==5, 'Input tensor does not have dimensionality of 5 [num_MC, Batch_Size, Features]'
		assert self.num_features==x.shape[2]

		if self.momentum is None:
			exponential_average_factor = 0.0
		else:
			exponential_average_factor = self.momentum

		if self.training and self.track_running_stats:
			# TODO: if statement only here to tell the jit to skip emitting this when it is None
			if self.num_batches_tracked is not None:
				self.num_batches_tracked += 1
				if self.momentum is None:  # use cumulative moving average
					exponential_average_factor = 1.0 / float(self.num_batches_tracked)
				else:  # use exponential moving average
					exponential_average_factor = self.momentum

		out = x.permute(1,2,3,4,0) # shape = [batch_size, features, MC] since batchnorm computes statistics for second dimension
		out = F.batch_norm(out, self.running_mean, self.running_var, self.weight, self.bias,
		                   self.training or not self.track_running_stats,
		                   exponential_average_factor, self.eps)
		out = out.permute(4,0,1,2,3) # shape = [MC, batch_size, features]
		return out

#Expansion Layers
class MC_ExpansionLayer(torch.nn.Module):

	def __init__(self, num_MC=1, input_dim=2):
		'''

		:param num_MC: if input.dim()==input_dim, expand first dimension by num_MC
		:param input_dim: determines un-expanded input dim
		'''
		super().__init__()

		self.num_MC = num_MC
		self.input_dim = input_dim

	def forward(self, x):

		if x.dim()==self.input_dim:
			out = x.unsqueeze(0).repeat(self.num_MC, *(x.dim() * (1,)))
		elif x.dim()==self.input_dim+1:
			out = x
		else:
			raise ValueError(f"Input.dim()={x.dim()}, but should be either {self.input_dim} and expanded or {self.input_dim+1}")

		return out

#Gaussian Variational Distributions
class VariationalNormal(torch.nn.Module):

	def __init__(self, loc, scale):

		super().__init__()

		self.loc 	= torch.nn.Parameter(loc)
		self.logscale 	= torch.nn.Parameter(torch.log(torch.exp(scale)-1))

	def dist(self):

		return Normal(self.loc, F.softplus(self.logscale))

	def rsample(self, shape):

		if hasattr(self, 'samples'):
			del self.samples

		self.samples = self.dist().rsample(shape)

		self.samples.requires_grad_()
		self.samples.retain_grad()

		return self.samples

#Parallel Bayesian Sampling layers
class BayesLinear(torch.nn.Module):

	def __init__(self, in_features, out_features, num_MC=None, prior=1.):

		super().__init__()

		self.dim_input = in_features
		self.dim_output = out_features
		self.num_MC = num_MC

		self.mu_init_std = torch.sqrt(torch.scalar_tensor(2/(in_features + out_features)))
		# self.logsigma_init_std = torch.log(torch.exp(torch.scalar_tensor(self.mu_init_std))-1) #+ params.init_std_offset


		self.weight = VariationalNormal(FloatTensor(in_features, out_features).normal_(0., self.mu_init_std),
		                                FloatTensor(in_features, out_features).fill_(self.mu_init_std))

		self.bias = VariationalNormal(FloatTensor(out_features).normal_(0., self.mu_init_std),
		                              FloatTensor(out_features).fill_(self.mu_init_std))

		if prior=='laplace':
			self.prior = LaplacePrior(module=self)
		elif prior=='laplace_clamp':
			self.prior = LaplacePrior(module=self, clamp=True)
		elif isinstance(float(prior), Number):
			self.prior = GaussianPrior(scale=prior)
		else:
			exit('Wrong Prior ... should be in [1.0, "laplace"]')

		self.reset_parameters(scale_offset=+1)

	def reset_parameters(self, scale_offset=0):

		torch.nn.init.kaiming_uniform_(self.weight.loc.data, a=math.sqrt(5))
		self.weight.logscale.data.fill_(torch.log(torch.exp((self.mu_init_std)/self.weight.loc.shape[1] )-1)+scale_offset)
		if self.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.loc.data)
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias.loc.data, -bound, bound)
			self.bias.logscale.data.fill_(torch.log(torch.exp(self.mu_init_std)-1)+scale_offset)

	def forward(self, x : torch.Tensor, prior=None, stochastic=True):
		'''

		:param x: 3 dimensional tensor of shape [N_MC, M_BatchSize, d_Input]
		:return:
		'''

		assert x.dim()==3, f"Input tensor not of shape [N_MC, BatchSize, Features] but is {x.shape=}"
		# assert x.shape[0]==self.num_MC

		num_MC = x.shape[0]
		bs = x.shape[1]

		forward = ['reparam', 'local_reparam'][0]

		if forward=='reparam':
			# Standard Reparam Trick

			# self.locs = self.weight.loc.unsqueeze(0).repeat(num_MC, 1, 1)
			# self.logscales = self.weight.logscale.unsqueeze(0).repeat(num_MC, 1, 1)
			#
			# self.sampled_w = self.locs + torch.ones_like(self.locs).normal_() * F.softplus(self.logscales)
			#
			# self.locs.requires_grad_()
			# self.locs.retain_grad()
			# self.logscales.requires_grad_()
			# self.logscales.retain_grad()
			# self.sampled_w.requires_grad_()
			# self.sampled_w.retain_grad()
			# print(self.sampled_w)

			# print(f"BayesLinear.forward: {x.shape=} {self.sampled_w.shape=} {self.sampled_b.shape=}")

			# out = torch.baddbmm(b, x, w)

			# self.weight.logscale.data.fill_(0.0001)

			self.sampled_w = self.weight.rsample((num_MC,))
			self.sampled_b = self.bias.rsample((num_MC,))
			# self.sampled_w.requires_grad_(); self.sampled_w.retain_grad()
			# self.sampled_b.requires_grad_();self.sampled_b.retain_grad()
			# self.sampled_w.requires_grad_()
			# self.sampled_w.retain_grad()

			# print(f"{self.sampled_w.shape=}")
			# exit()
			out = torch.baddbmm(self.sampled_b.unsqueeze(1), x, self.sampled_w)
			# out = torch.baddbmm(self.bias.loc.unsqueeze(0).repeat(num_MC,1,1), x, self.weight.loc.unsqueeze(0).repeat(num_MC,1,1))
			# out = torch.bmm(x, self.sampled_w)

		if forward=='local_reparam':
			# LOCAL REPARAM TRICK
			w_sigma = F.softplus(self.weight_logscale)
			mean = torch.matmul(x, self.weight_loc) + self.bias_loc
			std = torch.sqrt(torch.matmul(x.pow(2), F.softplus(self.weight_logscale).pow(2)) + F.softplus(self.bias_logscale).pow(2))
			# std = torch.matmul(x.pow(2), F.softplus(self.w_logsigma).pow(2)) + F.softplus(self.b_logsigma).pow(2) # Mathematically not correct, gives extreme uncertainties if not regularized
			# print(mean.shape, std.shape, x.shape)
			epsilon = FloatTensor(x.shape[0], x.shape[1], self.dim_output).normal_(0., self.epsilon_sigma)
			# print(mean.shape, epsilon.shape, std.shape)
			out = mean + epsilon*std


			# prior_mean = torch.matmul(x, FloatTensor(*self.w_mu.shape).fill_(0.)) + self.b_mu
			# prior_std = 1
			# if prior is not None: prior = prior + self.analytic_prior(_sigma=w_sigma, _sigma_prior=self.sigma_prior, _mu=self.w_mu, _mu_prior=FloatTensor([0.]))

		self.kl_div = torch.distributions.kl_divergence(self.weight.dist(), self.prior.dist()).sum()
		self.entropy = self.weight.dist().entropy().sum()

		return out

class BayesConv2d(torch.nn.Module):

	# NegHalfLog2PI = -.5 * torch.log(2.0 * FloatTensor([np.pi]))

	def __init__(self, in_channels, out_channels, kernel_size, stride=1, num_MC=1, prior_scale=1):
		'''

		:param in_channels:
		:param out_channels:
		:param kernel_size:

		Kernelsize: [out_channel, in_channel, kernel_size, kernel_size]
		'''

		super().__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.epsilon_sigma = 1.0
		self.num_MC = num_MC
		self.prior_scale = prior_scale

		self.mu_init_std = torch.scalar_tensor(1./np.sqrt(in_channels*kernel_size**2))
		# self.logsigma_init_std = torch.scalar_tensor(np.log(np.exp(self.mu_init_std)-1)) #+ params.init_std_offset

		self.weight = VariationalNormal(FloatTensor(out_channels, in_channels, kernel_size, kernel_size).normal_(0, self.mu_init_std),
		                                FloatTensor(out_channels, in_channels, kernel_size, kernel_size).fill_(self.mu_init_std))

		self.bias = VariationalNormal(FloatTensor(out_channels).normal_(0., self.mu_init_std),
		                              FloatTensor(out_channels).fill_(self.mu_init_std))

		self.reset_parameters(scale_offset=0.)


	def reset_parameters(self, scale_offset=0):

		torch.nn.init.kaiming_uniform_(self.weight.loc.data, a=math.sqrt(5))
		self.weight.logscale.data.fill_(torch.log(torch.exp((self.mu_init_std+scale_offset)/self.weight.loc.shape[1] )-1))
		# self.weight.logscale.data.fill_(torch.log(torch.exp(self.mu_init_std)-1)+scale_offset)
		if self.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.loc.data)
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias.loc.data, -bound, bound)
			self.bias.logscale.data.fill_(torch.log(torch.exp(self.mu_init_std + scale_offset)-1))

	def forward(self, x : torch.Tensor, stochastic=True, prior=None):
		'''
		torch.nn.functional.conv{1d, 2d, 3d} allows the use of groups
		Groups split the in_channels into a set of individual group in_channels which are processed with separate kernels
		We want to fold the MC dimension therefore into the channel dimension.
		'''


		assert x.dim()==5, 'Input tensor not of shape [num_MC, batch_size in_channels, height, width]'
		assert x.shape[0] == self.num_MC, f'Input.shape={x.shape} with {x.shape[0]}!={self.num_MC}'

		MC, BS, C, H, W = x.shape

		'''
		Move BS dimension to the front 
		Flatten MC and BS dimension into [BS, [ MC_C_1, MC_C_2, MC_C_3 ... ], H, W]
		
		'''
		out = x.permute(1,0,2,3,4) # [MC, BS, C, H, W] -> [BS, MC, C, H, W]
		out = out.flatten(1,2).contiguous() # [BS, MC * C, H, W]

		'''
		Sample MC number of weights and biases
		'''
		w = self.weight.rsample((MC,)) # [MC, C_Out, C_In, Kernel_H, Kernel_W]
		b = self.bias.rsample((MC,)) # [MC, C_Out]

		'''
		Flatten parameters into groups [ MC_Weight_1, MC_Weight_2, MC_Weight_3 ... ] for weights and biases
		'''
		w = w.flatten(0,1).contiguous()
		b = b.flatten().contiguous()

		'''
		Data: 	[BS, [MC_C_1, MC_C_2, MC_C_3, MC_C_4 ... ], H, W]
		Kernel: [MC_Weight_1, MC_Weight_2, MC_Weight_3
		'''
		out = F.conv2d(out, w, groups=MC, bias=b, stride=self.stride)

		'''
		Unsplit the MC dim from the C_Out dim and move MC dim again to the front
		'''
		out = out.reshape(BS, MC, self.out_channels, out.shape[-2], out.shape[-1]).contiguous()
		out = out.permute(1,0,2,3,4) # [MC, BS, C_Out, H, W]

		self.kl_div = torch.distributions.kl_divergence(self.weight.dist(), Normal(0, self.prior_scale)).sum()

		return out

class BayesianNeuralNetwork(torch.nn.Module):

	def __init__(self, num_MC):

		super().__init__()

		self.num_MC = num_MC

	def forward(self, x):

		x = x.unsqueeze(0).repeat(self.num_MC, *(x.dim() * (1,)))

		return x

	def collect_kl_div(self):
		'''
		self.kl_div = 0

		for name, module in self.named_children():

			# print(f'@kl_div {any([isinstance(module, layer) for layer in [BayesLinear, BayesConv2d]])}')

			if any([isinstance(module, layer) for layer in [BayesLinear, BayesConv2d]]):
				self.kl_div = self.kl_div + module.kl_div

		return self.kl_div
		'''

		raise NotImplementedError

if __name__=="__main__":

	if True: # testing MC_BatchNorm1D
		num_samples = 1000
		data = torch.distributions.Normal(Tensor([1.,1.,1.]), Tensor([1.0, 5., 0.1])).sample((num_samples,))
		data = data.reshape((num_samples//5,5,3))

		target = torch.distributions.Normal(Tensor([-3.,3.,5.]), Tensor([1., 10., 1])).sample((num_samples,))
		target = target.reshape((num_samples//5,5,3))

		# data = Tensor([1,1,1]).reshape(1,1,3).repeat(20,5,1)
		# target = Tensor([1,2,3]).expand_as(data)

		# print(data.shape)
		# exit()

		# bn = torch.nn.BatchNorm1d(num_features=3)
		# data, target = data[0], target[0]

		bn = MC_BatchNorm1d(num_features=3)
		# bn.bias.data = Tensor([10,10,10])

		optim = torch.optim.Adam(bn.parameters(), lr=0.01)
		bn.train()
		num_epochs = 2000
		for epoch in range(num_epochs):
			optim.zero_grad()
			# bn.bias.data = torch.zeros_like(bn.bias.data)
			pred = bn(data)
			loss = F.mse_loss(pred, target)
			loss.backward()
			optim.step()
			if epoch%(num_epochs//5)==0: print(f'loss {loss.item():.2f} bias:{bn.bias.data}, weight {bn.weight.data}')

		print(f'bias:{bn.bias.data}, weight {bn.weight.data}')
		bn.eval()
		print(bn(data).mean(dim=0))
		print(bn(data).std(dim=0))
# if True:
# 	data = torch.distributions.Normal(Tensor([1.,1.,1.]), Tensor([1.0, 5., 0.1])).sample((num_samples,))