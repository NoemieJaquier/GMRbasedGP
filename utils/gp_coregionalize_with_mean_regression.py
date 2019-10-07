#! /usr/bin/env python
##################################################
# Author: Noemie Jaquier, 2019
# License: MIT
# Contact: noemie.jaquier@idiap.ch
##################################################

import numpy as np
from GPy.core import GP
from GPy import kern
from GPy import util


class GPCoregionalizedWithMeanRegression(GP):
	"""
	Gaussian Process model for heteroscedastic multioutput regression with a prior mean

	This is a thin wrapper around the models.GP class, with a set of sensible defaults

	:param X_list: list of input observations corresponding to each output
	:type X_list: list of numpy arrays
	:param Y_list: list of observed values related to the different noise models
	:type Y_list: list of numpy arrays
	:param kernel: a GPy kernel ** Coregionalized, defaults to RBF ** Coregionalized
	:type kernel: None | GPy.kernel defaults
	:likelihoods_list: a list of likelihoods, defaults to list of Gaussian likelihoods
	:type likelihoods_list: None | a list GPy.likelihoods
	:param name: model name
	:type name: string
	:param W_rank: number tuples of the corregionalization parameters 'W' (see coregionalize kernel documentation)
	:type W_rank: integer
	:param kernel_name: name of the kernel
	:type kernel_name: string

	"note: adapted from GPy function "GPCoregionalizedRegression(GP)"
	"""

	def __init__(self, X_list, Y_list, kernel=None, likelihoods_list=None, mean_function = None, name='GPCR', W_rank=1, kernel_name='coreg'):
		# Input and Output
		X, Y, self.output_index = util.multioutput.build_XY(X_list, Y_list)
		Ny = len(Y_list)

		# Kernel
		if kernel is None:
			kernel = kern.RBF(X.shape[1] - 1)

			kernel = util.multioutput.ICM(input_dim=X.shape[1] - 1, num_outputs=Ny, kernel=kernel, W_rank=1, name=kernel_name)

		# Likelihood
		likelihood = util.multioutput.build_likelihood(Y_list, self.output_index, likelihoods_list)

		super(GPCoregionalizedWithMeanRegression, self).__init__(X, Y, kernel, likelihood, mean_function, Y_metadata={'output_index': self.output_index})