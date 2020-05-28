import torch
from torch.distributions import Normal, Dirichlet, Categorical, RelaxedOneHotCategorical
from torch.nn import Sequential, Tanh, ReLU, Linear, Dropout, CELU, BatchNorm1d, Parameter
import torch.nn.functional as F

import numpy as np
import math
import matplotlib.pyplot as plt

torch.manual_seed(1234)

if torch.cuda.is_available():
	FloatTensor = torch.cuda.FloatTensor
	Tensor = torch.cuda.FloatTensor
elif not torch.cuda.is_available():
	FloatTensor = torch.FloatTensor
	Tensor = torch.FloatTensor

#Activation functions
class ShiftedReLU(torch.nn.Module):

	def __init__(self, offset=-1):
		super().__init__()

		self.offset = offset

	def forward(self, _x):
		out = torch.clamp(input=_x, min=self.offset)

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

	def forward(self, _x):
		out = torch.clamp(input=_x, min=self.offset) + self.slope * _x * _x.le(
			self.offset).float() - self.slope * self.offset * _x.le(self.offset).float()

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

def MC_dropout1d(_x, _p, _num_MC=None):

	assert _x.dim() == 4, 'Input tensor does not have dimensionality of 5 [num_MC, Batch_Size, Ci, H, W]'
	if _num_MC != None: assert _x.shape[0]==_num_MC

	num_MC = _x.shape[0]
	batch_size = _x.shape[1]
	dim_out = _x.shape[2]

def MC_dropout2d(_x, _p, _num_MC=None):

	assert _x.dim() == 5, 'Input tensor does not have dimensionality of 5 [num_MC, Batch_Size, Ci, H, W]'
	if _num_MC != None: assert _x.shape[0]==_num_MC

	num_MC = _x.shape[0]
	batch_size = _x.shape[1]
	dim_out = _x.shape[2]

def MC_CrossEntropy(pred, target):
	assert pred.dim()==3, f'Prediction should be of shape [MC, BS, C] but is only of shape {pred.shape}'
	assert target.dim()==1, f'Targets should be of shape [BS] with each correct label but is only of shape {target.shape}'

	num_MC, batch_size, num_C = pred.shape

	MC_target = FloatTensor(batch_size, num_C).fill_(0)
	MC_target.scatter_(1, target.unsqueeze(-1), 1)
	MC_target = MC_target.unsqueeze(0).repeat((num_MC, 1, 1))
	# loss = -torch.sum(MC_target*F.log_softmax(pred, dim=-1))#/(num_MC*batch_size)
	# print(pred.shape, MC_target.shape)
	# print(MC_target[:,0,:])
	# print(MC_target.flatten(0,1).shape)
	# print(pred.flatten(0,1).shape)

	loss = F.cross_entropy(pred.flatten(0,1), target.unsqueeze(0).repeat(num_MC, 1).flatten(0,1).long())
	return loss

# SIVI Layers:
class SIVILayer(torch.nn.Module):

	def init_weights(self, _module):
		if type(_module) == torch.nn.Linear:
			torch.nn.init.orthogonal_(_module.weight, gain=1.)
			if type(_module.bias) != None:
				_module.bias.data.normal_(0., 0.01)

	def __init__(self, _dim_input, _dim_output, _dim_noise_input):

		super().__init__()

		self.dim_input = _dim_input
		self.dim_output = _dim_output
		self.dim_noise_input = _dim_noise_input

		self.dim_output_params = _dim_input * _dim_output * 2 + _dim_output * 2
		self.num_hidden = np.min([self.dim_output_params, int((_dim_noise_input + self.dim_output_params) / 2)])

		self.prior_sigma = torch.scalar_tensor(1.0)

		self.sivi_net = Sequential(Linear(self.dim_noise_input, self.num_hidden),
		                           Tanh(),
		                           # Linear(self.num_hidden, self.num_hidden),
		                           # Tanh(),
		                           Linear(self.num_hidden, self.dim_output_params))  # weight matrix x mu x logsigma + bias x mu x logsigma

		self.noise_dist = Normal(loc=torch.zeros((self.dim_noise_input,)), scale=torch.ones((self.dim_noise_input,)))
		self.sivi_net.apply(self.init_weights)

	def forward(self, _x: torch.Tensor, _prior):
		'''

		:param _x: 3 dimensional tensor of shape [N_MC, M_BatchSize, d_Input]
		:return:
		'''

		assert _x.dim() == 3, 'Input tensor not of shape [N_MC, BatchSize, Features]'
		# assert _x.shape[0]==_base_noise.shape[0], 'Input and base_noise should have the same number of num_MC samples'
		# assert _base_noise.shape[1]==self.dim_noise_input

		num_MC = _x.shape[0]
		batch_size = _x.shape[1]

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

		out = torch.bmm(_x, dist_w.rsample()) + dist_b.rsample().unsqueeze(1)

		prior_w = torch.distributions.kl_divergence(dist_w, Normal(torch.zeros_like(self.w_mu), self.prior_sigma * torch.ones_like(self.w_std)))
		prior_b = torch.distributions.kl_divergence(dist_b, Normal(torch.zeros_like(self.b_mu), self.prior_sigma * torch.ones_like(self.b_std)))

		prior = _prior + (torch.mean(prior_w) + torch.mean(prior_b)) / (num_MC * batch_size)

		return out, prior.squeeze()

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

				for i in range(w.shape[0]):
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

	def forward(self, _x: torch.Tensor, _prior, _base_noise):
		'''

		:param _x: 3 dimensional tensor of shape [N_MC, M_BatchSize, d_Input]
		:return:
		'''

		assert _x.dim() == 3, 'Input tensor not of shape [N_MC, BatchSize, Features]'
		assert _x.shape[0] == _base_noise.shape[0], 'Input and base_noise should have the same number of num_MC samples'
		assert _base_noise.shape[1] == self.dim_noise_input

		num_MC = _x.shape[0]
		batch_size = _x.shape[1]

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

		out = torch.bmm(_x, w) + b.unsqueeze(1)
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

	def forward(self, _x: torch.Tensor, _prior):

		assert _x.dim() == 5, 'Input tensor not of shape [Num_MC, BatchSize, Features, height, width]'

		num_MC = _x.shape[0]
		batch_size = _x.shape[1]
		out = _x.permute(1,0,2,3,4)
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

	def forward(self, _x: torch.Tensor, _prior, _base_noise):

		assert _x.dim() == 5, 'Input tensor not of shape [Num_MC, BatchSize, Features, height, width]'
		assert _base_noise.shape[0] == _x.shape[0], 'Base noise num_MC is different from _x num_MC'
		assert _base_noise.shape[1] == self.dim_noise_in, 'Base noise dim is different from predefined noise dim'


		num_MC = _x.shape[0]
		batch_size = _x.shape[1]
		out = _x.permute(1,0,2,3,4)
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

	def forward(self, _x):

		assert _x.dim()==3, 'Input tensor does not have dimensionality of 5 [num_MC, Batch_Size, Features]'
		assert self.num_features==_x.shape[-1]

		MC = _x.shape[0]
		BS = _x.shape[1]
		Features = _x.shape[2]

		# out = _x.permute(1,2,0) # shape = [batch_size, features, MC] since batchnorm computes statistics for second dimension
		out = _x.flatten(0,1).contiguous() # shape = [MC*batch_size, features]
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

	def forward(self, _x):

		assert _x.dim()==5, 'Input tensor does not have dimensionality of 5 [num_MC, Batch_Size, Features]'
		assert self.num_features==_x.shape[2]

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

		out = _x.permute(1,2,3,4,0) # shape = [batch_size, features, MC] since batchnorm computes statistics for second dimension
		out = F.batch_norm(out, self.running_mean, self.running_var, self.weight, self.bias,
		                   self.training or not self.track_running_stats,
		                   exponential_average_factor, self.eps)
		out = out.permute(4,0,1,2,3) # shape = [MC, batch_size, features]
		return out

class VariationalNormal(torch.nn.Module):

	def __init__(self, loc, scale):

		super().__init__()

		self.loc = torch.nn.Parameter(loc)
		self.logscale = torch.nn.Parameter(torch.log(torch.exp(scale)-1))
		# print(self.logscale)

		# exit()

	def dist(self):

		return Normal(self.loc, F.softplus(self.logscale))

	def rsample(self, shape):

		return self.dist().rsample(shape)

class VariationalGMM(torch.nn.Module):

	def __init__(self, locs, scales, num_components=None):
		'''

		:param weights: shape = R[batch_dim1, ..., batch_dimN, simplex]
		:param components: batched distribution where each parameter has shape = R[batch_dim1, ..., batch_dimN, simplex]
		:param validate_args:
		'''
		super().__init__()

		assert locs.shape==scales.shape
		assert locs.dim()>=2 and scales.dim()>=2
		assert locs.shape[-1]==num_components, f'{locs.shape=} != {num_components}'

		self.relaxedcategorical_temperature = 0.01

		self.dirichlet_concentration  = Parameter(torch.ones(*locs.shape[:-1], num_components)*10)
		self.locs                     = Parameter(locs)
		self.logscales                = Parameter(torch.log(torch.exp(scales)-1))

	# assert self.dirichlet_concentration.shape == self.components_loc.shape == self.components_logscale.shape

	def rsample(self, sample_shape):

		'''
		sample latent dist from last dimension
		num_MC is first dimension due to batchaddmm reasons
		A) permute locs&logscales to first dimension then torch.gather
		'''

		latent_dist = Dirichlet(F.softplus(self.dirichlet_concentration), validate_args=True).rsample(sample_shape).float()

		if self.training:
			# self.relaxedcategorical_temperature *= 0.99
			sampled_relaxed_cats  = RelaxedOneHotCategorical(temperature=self.relaxedcategorical_temperature, probs=latent_dist, validate_args=True).rsample()
			# sampled_relaxed_cats = F.softmax(1/self.relaxedcategorical_temperature * sampled_relaxed_cats, dim=-1)#.transpose(-1,0)
			# sampled_relaxed_cats = F.softmax(100 * sampled_relaxed_cats, dim=-1)#.transpose(-1,0)
			# sampled_cats = F.softmax()
		elif not self.training:
			# print('evaluating')
			# sampled_cats = Categorical(latent_dist, validate_args=True).sample() # shape = [sample_shape, batch_shape]
			sampled_relaxed_cats  = RelaxedOneHotCategorical(temperature=self.relaxedcategorical_temperature, probs=latent_dist, validate_args=True).rsample()
			# sampled_relaxed_cats = F.softmax(1/self.relaxedcategorical_temperature * sampled_relaxed_cats, dim=-1)#.transpose(-1,0)
			# sampled_relaxed_cats = F.softmax(100 * sampled_relaxed_cats, dim=-1)#.transpose(-1,0)


		# print(f'{latent_dist.shape=}, {sampled_cats.shape=}')

		# exit()
		# print(sampled_relaxed_cats[0,0,0])

		# sampled_relaxed_cats = F.softmax(1/self.relaxedcategorical_temperature *sampled_relaxed_cats, dim=-1)
		# print(sampled_relaxed_cats[:10])

		# print(f'{self.components_loc.shape=}, {sampled_relaxed_cats.shape=}')
		# sampled_loc = torch.gather(self.components_loc, dim=0, index=sampled_relaxed_cats)
		# sampled_scale = torch.gather(F.softplus(self.components_logscale), 0, sampled_relaxed_cats)

		samples = Normal(self.locs, F.softplus(self.logscales), validate_args=True).rsample(sample_shape)
		# samples = Normal(self.locs.unsqueeze(0).expand_as(sampled_relaxed_cats), F.softplus(self.components_logscale.unsqueeze(0).expand_as(sampled_relaxed_cats)), validate_args=True).rsample()
		# print(f'{samples.shape=}, {sampled_relaxed_cats.shape=}')
		# exit()
		samples = torch.sum(samples * sampled_relaxed_cats, dim=-1)
		# dims = [x for x in range(samples.dim())]
		# samples = samples.permute([dims[-1]]+dims[:-1])
		# print(f'{samples.shape=}')

		# exit()

		return samples

class BayesGMMLinear(torch.nn.Module):

	def __init__(self, in_features, out_features, num_components=7, num_MC=1, prior_scale=1):

		super().__init__()

		self.in_features = in_features
		self.out_features = out_features

		self.weight         = VariationalGMM(locs=torch.randn(in_features, out_features, num_components),
		                                     scales=torch.empty(in_features, out_features, num_components).fill_(.1),
		                                     num_components=num_components)
		self.bias           = VariationalGMM(locs=torch.randn(out_features, num_components),
		                                     scales=torch.empty(out_features, num_components).fill_(.1),
		                                     num_components=num_components)

	def forward(self, x):

		assert x.dim() == 3
		num_MC, batch_size, features = x.shape
		# num_MC = 1000
		w = self.weight.rsample((num_MC,))
		b = self.bias.rsample((num_MC,)).unsqueeze(1)

		# plt.hist(w.detach().squeeze(), density=True, bins=100)
		# plt.show(); exit()

		# print(f'{b.shape=} {x.shape=} {w.shape=}')
		out = torch.baddbmm(b, x, w)

		return out

#Parallel Bayesian Sampling layers
class BayesLinear(torch.nn.Module):

	def __init__(self, in_features, out_features, num_MC=1, prior_scale=1):

		super().__init__()

		self.dim_input = in_features
		self.dim_output = out_features
		self.epsilon_sigma = 1.0
		self.num_MC = num_MC
		self.prior_scale = prior_scale

		self.mu_init_std = torch.sqrt(torch.scalar_tensor(2/(in_features + out_features)))
		self.logsigma_init_std = torch.log(torch.exp(torch.scalar_tensor(self.mu_init_std))-1) -3 #+ params.init_std_offset

		# self.weight_loc = torch.nn.Parameter(FloatTensor(in_features, out_features).normal_(0., self.mu_init_std))
		# self.weight_logscale = torch.nn.Parameter(FloatTensor(in_features, out_features).normal_(self.logsigma_init_std, self.mu_init_std))

		self.weight = VariationalNormal(FloatTensor(in_features, out_features).normal_(0., self.mu_init_std),
		                                FloatTensor(in_features, out_features).fill_(self.mu_init_std))

		self.bias = VariationalNormal(FloatTensor(out_features).normal_(0., self.mu_init_std),
		                              FloatTensor(out_features).fill_(self.mu_init_std))

		# self.bias_loc = torch.nn.Parameter(FloatTensor(out_features).normal_(0, self.mu_init_std))
		# self.bias_logscale = torch.nn.Parameter(FloatTensor(out_features).normal_(self.logsigma_init_std, self.mu_init_std))

		# self.log_pw = FloatTensor([0.])
		# self.log_qw = FloatTensor([0.])

		# self.sigma_prior = FloatTensor([1]).expand_as(self.weight_logscale)
		# self.mu_prior = FloatTensor([0.]).expand_as(self.weight_logscale)



		self.reset_parameters(scale_offset=0)

		# exit()

	# Analytic prior: KL[s || t] = log (sigma_t / sigma_s) + (sigma_s**2 + (mu_s - mu_t)**2) / 2 sigma_t**2 - 0.5
	# Analytic prior: KL[q || p]

	def reset_parameters(self, scale_offset=0):

		torch.nn.init.kaiming_uniform_(self.weight.loc.data, a=math.sqrt(5))
		self.weight.logscale.data.fill_(torch.log(torch.exp((self.mu_init_std)/self.weight.loc.shape[1] )-1)+scale_offset)
		if self.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.loc.data)
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias.loc.data, -bound, bound)
			self.bias.logscale.data.fill_(torch.log(torch.exp(self.mu_init_std)-1)+scale_offset)

	def forward(self, _x : torch.Tensor, prior=None, stochastic=True):
		'''

		:param _x: 3 dimensional tensor of shape [N_MC, M_BatchSize, d_Input]
		:return:
		'''

		assert _x.dim()==3, 'Input tensor not of shape [N_MC, BatchSize, Features]'
		# assert _x.shape[0]==self.num_MC

		num_MC = _x.shape[0]
		batch_size = _x.shape[1]

		forward = 'reparam'

		if forward=='reparam':
			# Standard Reparam Trick

			# dist_w = Normal(self.weight_loc, stochastic*F.softplus(self.weight_logscale))
			# dist_b = Normal(self.bias_loc.unsqueeze(1), stochastic*F.softplus(self.bias_logscale).unsqueeze(1).repeat(1,batch_size, 1))
			# dist_b = Normal(self.bias_loc.unsqueeze(0).repeat(batch_size, 1), stochastic*F.softplus(self.bias_logscale).unsqueeze(0).repeat(batch_size, 1))
			# dist_b = Normal(self.bias_loc.unsqueeze(0).repeat(1, 1), stochastic*F.softplus(self.bias_logscale).unsqueeze(0).repeat(1, 1))

			# prior = (prior + torch.distributions.kl_divergence(dist_w, Normal(0,self.prior_scale)).sum()) if prior is not None else None

			# w = dist_w.rsample((num_MC,))
			# b = dist_b.rsample((num_MC,))

			w = self.weight.rsample((num_MC,))
			b = self.bias.rsample((num_MC,1,))

			# print(w.shape, w2.shape, b.shape, b2.shape)
			# exit()
			prior = (prior + torch.distributions.kl_divergence(self.weight.dist(), Normal(0,self.prior_scale)).sum()) if prior is not None else None

			# print(b.shape, _x.shape, w.shape)
			# exit()
			out = torch.baddbmm(b, _x, w)

			# print(self.weight.logscale)

			# print(out)
			# print(out2)

			# exit()

		if forward=='local_reparam':
			# LOCAL REPARAM TRICK
			w_sigma = F.softplus(self.weight_logscale)
			mean = torch.matmul(_x, self.weight_loc) + self.bias_loc
			std = torch.sqrt(torch.matmul(_x.pow(2), F.softplus(self.weight_logscale).pow(2)) + F.softplus(self.bias_logscale).pow(2))
			# std = torch.matmul(_x.pow(2), F.softplus(self.w_logsigma).pow(2)) + F.softplus(self.b_logsigma).pow(2) # Mathematically not correct, gives extreme uncertainties if not regularized
			# print(mean.shape, std.shape, _x.shape)
			epsilon = FloatTensor(_x.shape[0], _x.shape[1], self.dim_output).normal_(0., self.epsilon_sigma)
			# print(mean.shape, epsilon.shape, std.shape)
			out = mean + epsilon*std


			# prior_mean = torch.matmul(_x, FloatTensor(*self.w_mu.shape).fill_(0.)) + self.b_mu
			# prior_std = 1
			# if prior is not None: prior = prior + self.analytic_prior(_sigma=w_sigma, _sigma_prior=self.sigma_prior, _mu=self.w_mu, _mu_prior=FloatTensor([0.]))

		return out, prior


	def analytic_prior(self, _sigma : torch.Tensor, _sigma_prior : torch.Tensor, _mu : torch.Tensor, _mu_prior : torch.Tensor):

		'''
		KL[q || p] = log[sigma_p / sigma_q] + ( sigma_q**2 + (mu_q - mu_p)**2 ) / 2 sigma_p**2 - 0.5
		:param sigma: variational sigma
		:param prior_sigma:
		:param mu: variational sigma
		:param prior_mu:
		:return:
		'''
		# print('analytic prior', torch.sum(_sigma), torch.sum(_sigma_prior), torch.sum(_mu))
		prior = torch.log(_sigma/_sigma_prior) + (_sigma**2 + (_mu - _mu_prior)**2 ) / 2*_sigma_prior**2 - 0.5
		prior = torch.sum(prior)
		# print(prior)
		# print()
		return prior

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

		self.reset_parameters(scale_offset=1.0)


	def reset_parameters(self, scale_offset=0):

		torch.nn.init.kaiming_uniform_(self.weight.loc.data, a=math.sqrt(5))
		self.weight.logscale.data.fill_(torch.log(torch.exp((self.mu_init_std+scale_offset)/self.weight.loc.shape[1] )-1))
		# self.weight.logscale.data.fill_(torch.log(torch.exp(self.mu_init_std)-1)+scale_offset)
		if self.bias is not None:
			fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight.loc.data)
			bound = 1 / math.sqrt(fan_in)
			torch.nn.init.uniform_(self.bias.loc.data, -bound, bound)
			self.bias.logscale.data.fill_(torch.log(torch.exp(self.mu_init_std + scale_offset)-1))

	def forward(self, _x : torch.Tensor, stochastic=True, prior=None):
		'''

		:param _x: 5 dimensional tensor of shape [BatchSize, N_MC x in_channels, height, width]
		:return:
		MC Kernelsize: [out_channel, in_channel, kernel_size, kernel_size]
		'''

		# assert _x.dim()==5, 'Input tensor not of shape [BatchSize, num_MC, in_channels, height, width]' old one
		assert _x.dim()==5, 'Input tensor not of shape [num_MC, batch_size in_channels, height, width]'
		assert _x.shape[0] == self.num_MC, f'Input.shape={_x.shape} with {_x.shape[0]}!={self.num_MC}'

		num_MC = _x.shape[0]
		batch_size = _x.shape[1]
		out = _x.permute(1,0,2,3,4) # shape = [ batch_size, num_MC, dim_in, height, width ]
		# print('x', _x.shape)
		out = out.flatten(1,2).contiguous()
		# print('conv input', out.shape)

		# dist_w = Normal(self.weight_loc, stochastic*F.softplus(self.weight_logscale))
		# dist_b = Normal(self.bias_loc, stochastic*F.softplus(self.bias_logscale))

		# prior = (prior + torch.distributions.kl_divergence(dist_w, Normal(0,self.prior_scale)).sum()) if prior is not None else None
		#
		# w = dist_w.rsample((num_MC,))
		# b = dist_b.rsample((num_MC,))

		w = self.weight.rsample((num_MC,))
		b = self.bias.rsample((num_MC,))

		prior = (prior + torch.distributions.kl_divergence(self.weight.dist(), Normal(0,self.prior_scale)).sum()) if prior is not None else None

		w = w.flatten(0,1).contiguous().contiguous()
		b = b.permute(1,0).flatten().contiguous()

		out = F.conv2d(out, w, groups=num_MC, bias=b, stride=self.stride)


		out = out.reshape(batch_size, num_MC, self.out_channels, out.shape[-2], out.shape[-1]).contiguous()
		out = out.permute(1,0,2,3,4)

		return out, prior

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