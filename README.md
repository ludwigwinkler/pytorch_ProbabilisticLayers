# torch-ProbabilisticLayers

<img src="data/plots/BNN.gif" width="500">

This repository implements **parallelized** Bayesian Neural Networks in PyTorch via Variational Inference.

Bayesian neural networks require the evaluation of the evidence lower bound as the cost function of choice which includes the expectation over the data log-likelihood.
\
The sampling of the data log-likelihood evidence is implemented in a parallel fashion to circumvent slow Python loops like in other repositories.

In order to achieve this, the most common layers are implemented from scratch to process the samples of the expectation of the data log-likelihood in parallel.
\
The data tensors fed into the Bayesian neural network are extended with an additional first dimension `input.shape=(MonteCarloSamples, BatchSize, Features...)` which is referred to as MC (Monte Carlo) dimension.

Caveat Emptor: The memory footprint increases linearly with the number of chosen Monte Carlo samples but the gradients are greatly stabilized even with > 5 MC samples.