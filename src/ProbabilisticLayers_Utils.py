import future, sys, os, datetime, argparse, time
# print(os.path.dirname(sys.executable))
from dataclasses import dataclass
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

import scipy
import scipy as sp
from scipy.io import loadmat as sp_loadmat
import copy

'''
Utils.Utils
'''


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')


class AverageMeter:

	def __init__(self):
		'''
		vals: a nested list epoch losses -> [epoch0[0.99, 0.98 ..., 0.54], epoch1[0.54, ... 0.25], epoch2[...]]

		'''
		self.vals = []
		self.num_batches = 0
		self.num_epoch_batches = 0

		self.reset()

	def reset(self):
		self.vals = []
		self.num_batches = 0
		self.num_epoch_batches = 0

	def epoch_reset(self):

		self.vals.append([])
		self.num_epoch_batches = 0

	def update(self, val):

		self.vals[-1].append(val)
		self.num_epoch_batches += 1
		self.num_batches += 1

	@property
	def avgs(self):

		vals = [val for epoch_vals in self.vals for val in epoch_vals]

		if len(vals) == 0:
			return [-1]
		elif len(vals) > 0:
			return np.around(vals, 3)

	@property
	def epoch_avg(self):

		if len(self.vals) == 0:
			''' Happens when no data has been fed yet'''
			return -1
		elif len(self.vals[-1]) > 0:
			return np.around(sum(self.vals[-1]) / len(self.vals[-1]), 3)

	@property
	def epoch_avgs(self):

		print(f'@epoch_avgs')
		print(self.vals)

		if len(self.vals) == 0:
			return -1
		elif len(self.vals) > 0:
			return [np.around(sum(epoch_vals) / len(epoch_vals), 3) for epoch_vals in self.vals]


class RunningAverageMeter(object):
	"""Computes and stores the average and current value"""

	def __init__(self, momentum=0.99, _max_non_improving=20, init_avg=1.0):
		self.momentum = momentum
		self.last_value = 0
		self.non_improving_counter = 0
		self.step = 0
		self.max_non_improving_steps = _max_non_improving
		self.best_val = 10000

		self.avg = init_avg

	def reset(self):
		self.val = None
		self.last_val = None
		self.avg = None
		self.last_avg = None
		self.step = 0

	def update(self, val):

		self.last_val = val
		self.step += 1

		if self.avg is None:
			self.avg = val
			self.last_avg = self.avg
		else:
			self.last_avg = self.avg
			self.avg = self.avg * self.momentum + val * (1 - self.momentum)

	def stopping_criterion(self):

		if self.avg <= self.best_val:
			self.best_val = self.avg
			self.non_improving_counter = 0
		elif self.avg >= self.best_val:
			self.non_improving_counter += 1

		if self.non_improving_counter >= self.max_non_improving_steps:
			return True
		else:
			return False


class Timer:
	"""Record multiple running times."""

	def __init__(self):
		self.times = []
		self.start()

	def start(self):
		self.tik = time.time()

	def stop(self):
		# Stop the timer and record the time in a list
		self.times.append(time.time() - self.tik)
		return self.times[-1]

	def avg(self):
		# Return the average time
		return sum(self.times) / len(self.times)

	def sum(self):
		# Return the sum of time
		return sum(self.times)

	def cumsum(self):
		# Return the accumulated times
		return np.array(self.times).cumsum().tolist()


class Benchmark:
	def __init__(self, description='Done in %.4f sec', repetitions=1):
		'''
		repetitions = 3
		with Benchmark():
			 for _ in range(3):
			 	run_some_code_xyz()
		-> prints descriptions with running_time/repetitions
		'''
		self.description = description
		self.repetitions = repetitions

	def __enter__(self):
		self.timer = Timer()
		return self

	def __exit__(self, *args):
		print(self.description % (self.timer.stop() / self.repetitions))

