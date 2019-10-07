#! /usr/bin/env python
##################################################
# Author: Noemie Jaquier, 2019
# License: MIT
# Contact: noemie.jaquier@idiap.ch
##################################################

import numpy as np
from GPy.core.mapping import Mapping
from GPy.core.parameterization import Param

class GmrMeanMapping(Mapping):
	"""
	Mean function for intrinsic/linear coregionalization GMR-based models

	:param gmr_model: gmr_model (from gmr.py or gmr_manifold_spd.py)

	"note: adapted from GPy function "MeanMapping(Mapping)"
	"""

	def __init__(self, input_dim, output_dim, gmr_model, name='gmrmeanmap'):
		super(GmrMeanMapping, self).__init__(input_dim=input_dim, output_dim=output_dim, name=name)
		self.gmr_model = gmr_model
		self.input_dim = input_dim

	def f(self, X):
		X = X[X[:, -1] == 0, 0:int((self.input_dim-1)/2)]
		output_dim = len(self.gmr_model.out_idx)
		nb_data = X.shape[0]
		exp_data = np.zeros((nb_data, output_dim))
		for i in range(nb_data):
			exp_data[i] = self.gmr_model.gmr_predict_mean(X[i, :])

		return np.hstack((exp_data[:, i] for i in range(output_dim)))[:, None]

	def update_gradients(self, dL_dF, X):
		return

	def gradients_X(self, dL_dF, X):
		return np.zeros(X.shape)
