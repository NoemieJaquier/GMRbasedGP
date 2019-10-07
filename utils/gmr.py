#! /usr/bin/env python
##################################################
# Author: Noemie Jaquier, 2019
# License: MIT
# Contact: noemie.jaquier@idiap.ch
# Those functions were implemented from pbdlib-python maintained by Emmanuel Pignat
# (https://gitlab.idiap.ch/rli/pbdlib-python)
##################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.linalg as sp_ln


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


class Gmr:
	def __init__(self, nb_states, nb_dim, in_idx, out_idx):
		"""
		Initialisation of Gmr class. Note: in this class, inputs and outputs are Euclidean (vectors).
		:param nb_states: number of GMM states
		:param nb_dim: number of dimensions of the data: input vector size + output vector size
		:param in_idx: input indexes
		:param out_idx: output indexes
		"""
		self.nb_states = nb_states
		self.nb_dim = nb_dim
		self.in_idx = in_idx
		self.out_idx = out_idx
		self.reg = 1e-8
		self.priors = None
		self.mu = None
		self.sigma = None  # covariance matrix
		self.inv_sigma = None  # Precision matrix


	# def init_params_kmeans(self, data):
	# 	from sklearn.cluster import KMeans
	# 	km_init = KMeans(n_clusters=self.nb_states)
	# 	km_init.fit(data)
	# 	self.mu_man = km_init.cluster_centers_
	# 	self.priors = np.ones(self.nb_states) / self.nb_states
	# 	self.sigma = np.array([np.eye(self.nb_dim) for i in range(self.nb_states)])

	def init_params_kbins(self, data, nb_samples):
		"""
		K_bins GMM initialisation
		:param data: np.array((nb_dim, nb_data))
		:param nb_samples: number of demonstrations
		:return: None
		"""

		# Delimit the cluster bins for first demonstration
		nb_data = int(data.shape[1]/nb_samples)

		self.priors = np.ones(self.nb_states)/self.nb_states
		self.mu = [np.zeros(self.nb_dim) for n in range(self.nb_states)]
		self.sigma = [np.zeros((self.nb_dim, self.nb_dim)) for n in range(self.nb_states)]

		t_sep = list(map(int, np.round(np.linspace(0, nb_data, self.nb_states + 1))))

		for i in range(self.nb_states):
			# Get bins indices for each
			inds = []
			for n in range(nb_samples):
				inds += range(n*nb_data + t_sep[i], n*nb_data + t_sep[i+1])
			data_tmp = data[:, inds]

			self.mu[i] = np.mean(data_tmp, axis=1)
			self.sigma[i] = np.cov(data_tmp) + np.eye(self.nb_dim) * self.reg

	def gmm_em(self, data, maxiter=100, minstepsize=1e-5):
		"""
		GMM computation with EM algorithm
		:param data: np.array((nb_dim, nb_data))
		:param maxiter: max number of iterations for EM
		:param minstepsize: maximum increase of log likelihood
		:return: likelihood vector
		"""

		nb_min_steps = 5  # min num iterations
		nb_max_steps = maxiter  # max iterations
		max_diff_ll = minstepsize  # max log-likelihood increase

		nb_data = data.shape[1]

		LL = np.zeros(nb_max_steps)
		for it in range(nb_max_steps):

			# E - step
			L = np.zeros((self.nb_states, nb_data))
			L_log = np.zeros((self.nb_states, nb_data))
			xts = [np.zeros((self.nb_dim, nb_data))] * self.nb_states

			for i in range(self.nb_states):
				L_log[i, :] = np.log(self.priors[i]) + multi_variate_normal(data.T, self.mu[i], self.sigma[i], log=True)

			L = np.exp(L_log)
			GAMMA = L / np.sum(L, axis=0)
			# GAMMA = L / (np.sum(L, axis=0) + 1e-300)
			GAMMA2 = GAMMA / (np.sum(GAMMA, axis=1)[:, np.newaxis])

			# M-step
			for i in range(self.nb_states):
				# Update Mu
				self.mu[i] = np.sum(data*GAMMA2[i], axis=1)

				# Update Sigma
				xtmp = data - self.mu[i][:, None]
				self.sigma[i] = np.dot(xtmp, np.dot(np.diag(GAMMA2[i]), xtmp.T)) + np.eye(self.nb_dim)*self.reg

			# Update priors
			self.priors = np.mean(GAMMA, axis=1)

			LL[it] = np.mean(np.log(np.sum(L, axis=0) + 1e-300))

			# Check for convergence
			if it > nb_min_steps:
				if LL[it] - LL[it - 1] < max_diff_ll:
					print('Converged after %d iterations: %.3e' % (it, LL[it]), 'red', 'on_white')
					return LL[it], GAMMA

		print("GMM did not converge before reaching max iteration. Consider augmenting the number of max iterations.")
		return LL[it], GAMMA

	def gmr_predict(self, input_data):
		"""
		GMR
		:param input_data: np_array(nb_dim), this function accept one input data at a time
		:return: expected mean, covariance and weights of the output data
		"""

		H = np.zeros(self.nb_states)

		# Compute weights
		for i in range(self.nb_states):
			H[i] = self.priors[i] * multi_variate_normal(input_data, self.mu[i][self.in_idx], self.sigma[i][self.in_idx][:, self.in_idx], log=False)
		H = H/(np.sum(H) + 1e-300)

		exp_data = np.zeros(len(self.out_idx))
		u_out = np.zeros((len(self.out_idx), self.nb_states))

		# Compute expected mean
		for i in range(self.nb_states):
			u_out[:, i] = self.mu[i][self.out_idx] + np.dot(np.dot(self.sigma[i][self.out_idx][:, self.in_idx], np.linalg.inv(self.sigma[i][self.in_idx][:, self.in_idx])), input_data - self.mu[i][self.in_idx])
			# Update expected mean
			exp_data += u_out[:, i]*H[i]

		# Compute expected covariance
		exp_cov = np.zeros((len(self.out_idx), len(self.out_idx)))
		exp_cov2 = np.zeros((len(self.out_idx), len(self.out_idx)))
		for i in range(self.nb_states):

			sigma_tmp = self.sigma[i][self.out_idx][:, self.out_idx] -	\
						np.dot(self.sigma[i][self.out_idx][:, self.in_idx], np.dot(np.linalg.inv(self.sigma[i][self.in_idx][:, self.in_idx]), self.sigma[i][self.in_idx][:,self.out_idx]))

			exp_cov += H[i] * (sigma_tmp + np.dot(u_out[:, i][:, None], u_out[:, i][None]))
			exp_cov2 += H[i] * (sigma_tmp + np.dot(u_out[:, i][:, None], u_out[:, i][None]) - np.dot(exp_data[:, None], exp_data[None]))

		exp_cov += - np.dot(exp_data[:, None], exp_data[None]) #+ np.eye(len(self.out_idx))*self.reg

		return exp_data, exp_cov, H

	def gmr_predict_mean(self, input_data):
		"""
		GMR (mean only)
		:param input_data: np_array(nb_dim), this function accept one input data at a time
		:return: expected mean of the output data
		"""

		H = np.zeros(self.nb_states)

		# Compute weights
		for i in range(self.nb_states):
			H[i] = self.priors[i] * multi_variate_normal(input_data, self.mu[i][self.in_idx], self.sigma[i][self.in_idx][:, self.in_idx], log=False)
		H = H/(np.sum(H) + 1e-300)

		exp_data = np.zeros(len(self.out_idx))
		u_out = np.zeros((len(self.out_idx), self.nb_states))

		# Compute expected mean
		for i in range(self.nb_states):
			u_out[:, i] = self.mu[i][self.out_idx] + np.dot(np.dot(self.sigma[i][self.out_idx][:, self.in_idx], np.linalg.inv(self.sigma[i][self.in_idx][:, self.in_idx])), input_data - self.mu[i][self.in_idx])
			# Update expected mean
			exp_data += u_out[:, i]*H[i]

		return exp_data


def plot_gmm(Mu, Sigma, color=[1, 0, 0], alpha=0.5, linewidth=1, markersize=6, ax=None, empty=False, edgecolor=None, edgealpha=None, border=False, center=True, zorder=20):
	"""
	This function displays the parameters of a Gaussian mixture model (GMM)
	:param Mu: centers of the Gaussians, np.array((nb_states, nb_dim))
	:param Sigma: covariance matrices of the Gaussians, np.array((nb_states, nb_dim, nb_dim))
	:param color: color of the displayed Gaussians
	:param alpha: transparency factor
	:param linewidth: width of the contours of the Gaussians
	:param markersize: size of the centers of the Gaussians
	:param ax: figure id
	:param empty: if true, plot wihout axis and grid
	:param edgecolor: color of the contour of the Gaussians
	:param edgealpha: transparency factor of the contours
	:param border: if true, plot points of the contours
	:param center: if true, plot the center of the Gaussian
	:param zorder:
	:return:

	Note: original function from Martijn Zeestraten, 2015
	"""

	nbStates = Mu.shape[0]
	nbDrawingSeg = 35
	t = np.linspace(-np.pi, np.pi, nbDrawingSeg)

	# if not isinstance(color, list) and not isinstance(color, np.ndarray):
	# 	color = [color] * nbStates
	# elif not isinstance(color[0], basestring) and not isinstance(color, np.ndarray):
	color = [color] * nbStates

	if not isinstance(alpha, np.ndarray):
		alpha = [alpha] * nbStates
	else:
		alpha = np.clip(alpha, 0.1, 0.9)

	for i, c, a in zip(range(0, nbStates), color, alpha):
		# Create Polygon
		R = np.real(sp_ln.sqrtm(1.0 * Sigma[i]))
		points = R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg])) + Mu[i][:, None]

		if edgecolor is None:
			edgecolor = c

		polygon = plt.Polygon(points.transpose().tolist(), facecolor=c, alpha=a, linewidth=linewidth, zorder=zorder, edgecolor=edgecolor)

		if edgealpha is not None:
			plt.plot(points[0, :], points[1, :], color=edgecolor)

		# if nb == 2:
		# 	R = np.real(sp.linalg.sqrtm(4.0 * Sigma[:, :, i]))
		# 	points = R.dot(np.array([[np.cos(t)], [np.sin(t)]]).reshape([2, nbDrawingSeg])) + Mu[:, i].reshape(2, 1])
		# 	polygon_2 = plt.Polygon(points.transpose().tolist(), facecolor=c, alpha=a / 2., linewidth=linewidth, zorder=zorder, edgecolor=edgecolor)

		if ax:
			# if nb == 2:
			# 	ax.add_patch(polygon_2)
			ax.add_patch(polygon)  # Patch

			l = None
			if center:
				a = alpha[i]
			else:
				a = 0.

			ax.plot(Mu[i][0], Mu[i][1], '.', color=c, alpha=a)  # Mean

			if border:
				ax.plot(points[0, :], points[1, :], color=c, linewidth=linewidth, markersize=markersize)  # Contour
		else:
			if empty:
				plt.gca().grid('off')
				# ax[-1].set_xlabel('x position [m]')
				plt.gca().set_axis_bgcolor('w')
				plt.axis('off')

			plt.gca().add_patch(polygon)  # Patch
			# if nb == 2:
			# 	ax.add_patch(polygon_2)
			# l = None

			if center:
				a = alpha[i]
			else:
				a = 0.0

			l, = plt.plot(Mu[i][0], Mu[i][1], '.', color=c, alpha=a)  # Mean

	return l
