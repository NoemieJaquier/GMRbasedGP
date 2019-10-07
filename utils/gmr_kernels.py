#! /usr/bin/env python
##################################################
# Author: Noemie Jaquier, 2019
# License: MIT
# Contact: noemie.jaquier@idiap.ch
##################################################

import numpy as np
from GPy.kern import Kern

from GPy.util.config import config # for assesing whether to use cython
try:
	from GPy import coregionalize_cython
	config.set('cython', 'working', 'True')
except ImportError:
	config.set('cython', 'working', 'False')


def multi_variate_normal(x, mu, sigma=None, log=True, inv_sigma=None):
	"""
	Multivariatve normal distribution PDF

	:param x:		np.array([nb_samples, nb_dim])
	:param mu: 		np.array([nb_dim])
	:param sigma: 	np.array([nb_dim, nb_dim])
	:param log: 	bool
	:return:
	"""
	dx = x - mu
	if sigma.ndim == 1:
		sigma = sigma[:, None]
		dx = dx[:, None]
		inv_sigma = np.linalg.inv(sigma) if inv_sigma is None else inv_sigma
		log_lik = -0.5 * np.sum(np.dot(dx, inv_sigma) * dx, axis=1) - 0.5 * np.log(np.linalg.det(2 * np.pi * sigma))
	else:
		inv_sigma = np.linalg.inv(sigma) if inv_sigma is None else inv_sigma
		log_lik = -0.5 * np.einsum('...j,...j', dx, np.einsum('...jk,...j->...k', inv_sigma, dx)) - 0.5 * np.log(np.linalg.det(2 * np.pi * sigma))

	return log_lik if log else np.exp(log_lik)


class GmrBasedCoregionalize(Kern):
	"""
	Covariance function for intrinsic/linear coregionalization GMR-based models

	This covariance has the form:
	.. math::
	   \mathbf{B} = h_\ell(\mathbf{x}_m) h_\ell(\mathbf{x}_n)\bm{\Sigma}_\ell^{Y}

	An intrinsic/linear coregionalization covariance function of the form:
	.. math::

	   k_2(x, y)=\mathbf{B} k(x, y)

	it is obtained as the tensor product between a covariance function
	k(x, y) and B.

	:param output_dim: number of outputs to coregionalize
	:type output_dim: int
	:param mu_gmr_in: input part of the means of GMM components
	:type mu_gmr_in: [np.array((input_dim)) for nb_states]
	:param sigma_gmr_in: input part of the covariances of GMM components
	:type sigma_gmr_in: [np.array((input_dim, input_dim)) for nb_states]
	:param priors_gmr: priors of GMM components
	:type priors_gmr: np.array((nb_states))
	:param sigma_gmr_out: output part of the covariances of GMM components
	:type sigma_gmr_out: [np.array((output_dim, output_dim)) for nb_states]
	:param gmr_index: index of the GMM component used for B
	:type gmr_index: int
	:param active_dims: active dimensions to compute B, range(input_dim, input_dim*2+1)
	:type active_dims: [int]

	"note: adapted from GPy function "Coregionalize(Kern)"

	"""
	def __init__(self, input_dim, output_dim, mu_gmr_in, mu_gmr_out, sigma_gmr_in, sigma_gmr_out_in, priors_gmr, sigma_gmr_cond, gmr_index, active_dims=None, name='gmrcoregion'):
		super(GmrBasedCoregionalize, self).__init__(input_dim, active_dims, name=name)

		# Parameters
		self.output_dim = output_dim
		self.mu_in = mu_gmr_in
		self.mu_out = mu_gmr_out
		self.sigma_in = sigma_gmr_in
		self.sigma_out_in = sigma_gmr_out_in
		self.priors_gmr = priors_gmr
		self.gmr_index = gmr_index
		self.nb_states = len(mu_gmr_in)
		self.sigma_out_cond = sigma_gmr_cond

		# Coregionalization parameters
		self.B = self.sigma_out_cond[self.gmr_index]
		self.W = None

	def parameters_changed(self):
		return

	def K(self, X, X2=None):
		# Compute weights
		W, h1, h2 = self._compute_weights(X, X2)

		# Compute coregionalized kernel
		if config.getboolean('cython', 'working'):
			return self._K_cython(self.B, X, X2) * W
		else:
			return self._K_numpy(self.B, X, X2) * W

	def _K_numpy(self, B, X, X2=None):
		X = X[:, -1][:, None]
		index = np.asarray(X, dtype=np.int)
		if X2 is None:
			return B[index,index.T]
		else:
			X2 = X2[:, -1][:, None]
			index2 = np.asarray(X2, dtype=np.int)
			return B[index,index2.T]

	def _K_cython(self, B, X, X2=None):
		X = X[:, -1][:, None]
		if X2 is None:
			return coregionalize_cython.K_symmetric(B, np.asarray(X, dtype=np.int64)[:,0])
		X2 = X2[:, -1][:, None]
		return coregionalize_cython.K_asymmetric(B, np.asarray(X, dtype=np.int64)[:,0], np.asarray(X2, dtype=np.int64)[:,0])


	def Kdiag(self, X):
		W = self._compute_weights(X)
		return np.diag(self.B)[np.asarray(X, dtype=np.int).flatten()] * W  # TODO check that it's right

	def update_gradients_full(self, dL_dK, X, X2=None):
		return

	def update_gradients_diag(self, dL_dKdiag, X):
		return

	def gradients_X(self, dL_dK, X, X2=None):
		return np.zeros(X.shape)

	def gradients_X_diag(self, dL_dKdiag, X):
		return np.zeros(X.shape)

	def _compute_weights(self, X, X2 = None):
		# Keep only one repetition and remove last column of X
		# (X is in the multi-output kernel format
		# Therefore it is composed of the input repeated D times. The last column is the index of the repetition.)
		X = X[X[:, -1] == 0, :-1]

		# Compute weights
		H = np.zeros((self.nb_states, X.shape[0]))
		for i in range(self.nb_states):
			H[i] = self.priors_gmr[i] * multi_variate_normal(X, self.mu_in[i], self.sigma_in[i], log=False)
		Haugm = np.hstack((H for i in range(self.output_dim)))
		Haugm = Haugm / (np.sum(Haugm, axis=0) + 1e-300)
		H1 = np.copy(Haugm[self.gmr_index])

		h1 = H / (np.sum(H, axis=0) + 1e-300)
		if X2 is None:
			H2 = H1
			h2 = H
		else:
			# Keep only one repetition and remove last column of X
			X2 = X2[X2[:, -1] == 0, :-1]

			# Compute weights
			H = np.zeros((self.nb_states, X2.shape[0]))
			for i in range(self.nb_states):
				H[i] = self.priors_gmr[i] * multi_variate_normal(X2, self.mu_in[i], self.sigma_in[i], log=False)
			Haugm = np.hstack((H for i in range(self.output_dim)))
			Haugm = Haugm / (np.sum(Haugm, axis=0) + 1e-300)
			H2 = np.copy(Haugm[self.gmr_index])

			h2 = H / (np.sum(H, axis=0) + 1e-300)

		# Compute weight matrix H1*H2'
		W = np.dot(H1[:, None], H2[:, None].T)

		return W, h1, h2


def Gmr_based_kernel(gmr_model, kernel_list, name='GMRbased'):
	"""
	GMR-based linear coregionalization kernel
	:param gmr_model: gmr model (gmr.py)
	:param kernel_list: list of GPy kernel with gmr_model.nb_states components
	:param name: name of the kernel
	:return: GPy GMR-based linear coregionalization kernel
	"""
	try:
		gmr_model.nb_states == len(kernel_list)
	except:
		print("You must provide one kernel per gmr state.")

	# Parameters
	input_dim = len(gmr_model.in_idx)
	output_dim = len(gmr_model.out_idx)
	active_dims = range(input_dim, input_dim * 2 + 1)

	# Mean and covariances
	mu_gmr_in = [gmr_model.mu[i][gmr_model.in_idx] for i in range(gmr_model.nb_states)]
	mu_gmr_out = [gmr_model.mu[i][gmr_model.out_idx] for i in range(gmr_model.nb_states)]
	sigma_gmr_in = [gmr_model.sigma[i][gmr_model.in_idx][:, gmr_model.in_idx] for i in range(gmr_model.nb_states)]
	sigma_gmr_out = [gmr_model.sigma[i][gmr_model.out_idx][:, gmr_model.out_idx] for i in range(gmr_model.nb_states)]
	sigma_gmr_out_in = [gmr_model.sigma[i][gmr_model.out_idx][:, gmr_model.in_idx] for i in range(gmr_model.nb_states)]

	sigma_gmr_cond = [
		sigma_gmr_out[i] - np.dot(sigma_gmr_out_in[i], np.dot(np.linalg.inv(sigma_gmr_in[i]), sigma_gmr_out_in[i].T))
		for i in range(gmr_model.nb_states)]

	# Initialisation of the kernel
	K = kernel_list[0].prod(GmrBasedCoregionalize(input_dim=input_dim + 1, output_dim=output_dim, mu_gmr_in=mu_gmr_in,
												  mu_gmr_out=mu_gmr_out,
												  sigma_gmr_in=sigma_gmr_in, sigma_gmr_out_in=sigma_gmr_out_in,
												  priors_gmr=gmr_model.priors,
												  sigma_gmr_cond=sigma_gmr_cond, gmr_index=0, active_dims=active_dims),
							name='%s%s' % (name, 0))

	# Kernel as a product of kernels associated to the GMM components
	j = 1
	for kernel in kernel_list[1:]:
		K += kernel.prod(GmrBasedCoregionalize(input_dim=input_dim + 1, output_dim=output_dim, mu_gmr_in=mu_gmr_in,
											   mu_gmr_out=mu_gmr_out,
											   sigma_gmr_in=sigma_gmr_in, sigma_gmr_out_in=sigma_gmr_out_in,
											   priors_gmr=gmr_model.priors,
											   sigma_gmr_cond=sigma_gmr_cond, gmr_index=j, active_dims=active_dims),
						 name='%s%s' % (name, j))
		j += 1

	return K
