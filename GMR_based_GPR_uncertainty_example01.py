#! /usr/bin/env python
##################################################
# Author: Noemie Jaquier, 2019
# License: MIT
# Contact: noemie.jaquier@idiap.ch
##################################################

import numpy as np
import GPy
import matplotlib.pyplot as plt
from utils.gmr import Gmr
from utils.gmr import plot_gmm
from utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
from utils.gmr_mean_mapping import GmrMeanMapping
from utils.gmr_kernels import Gmr_based_kernel


# Illustration of GMR-based GPR properties with 1-dimensional input and output
if __name__ == '__main__':

	# Parameters
	nb_data = 180
	input_dim = 1
	output_dim = 1
	nb_states = 2
	dt = 0.02

	# Parameters
	nb_obs = 0  # The user can change the number of observation to 0, 2 and 3.
	noise_var = 1e-4  # The user can change the noise variance.
	# noise_var = 0.01
	# noise_var = 0.1
	# lengthscale = 0.1  # The user can change the kernels lengthscales.
	lengthscale = 1.0
	# lengthscale = 5.0

	# Create artificial GMM
	gmr_model = Gmr(nb_states=nb_states, nb_dim=2, in_idx=[0], out_idx=[1])
	gmr_model.priors = np.ones(gmr_model.nb_states) / gmr_model.nb_states
	gmr_model.mu = [np.array([0.3, 1.4]), np.array([1.7, 0.4])]
	gmr_model.sigma = [np.array([[0.25, -0.07], [-0.07, 0.25]]), np.array([[0.08, 0], [0, 0.08]])]
	# gmr_model.mu = [np.array([0.3, 1.0]), np.array([2.0, 1.0])]
	# gmr_model.sigma = [np.array([[0.25, -0.0], [-0.0, 0.25]]), np.array([[0.08, 0], [0, 0.08]])]

	# Test data
	Xt = dt * np.arange(nb_data)[:, None] - dt*40.
	nb_data_test = Xt.shape[0]
	Xtest, _, output_index = GPy.util.multioutput.build_XY([np.hstack((Xt, Xt)) for i in range(output_dim)])

	# New observations (points to go through)
	if nb_obs is 0:
		X_obs = np.array([-20.0])[:, None]  # Ex 1
		Y_obs = np.array([1.0])
	elif nb_obs is 2:
		X_obs = np.array([0.3, 2.4])[:, None]  # Ex 2
		Y_obs = np.array([1.5, 0.3])
	elif nb_obs is 3:
		X_obs = np.array([0.3, 1.0, 2.4])[:, None]  # Ex 3
		Y_obs = np.array([1.5, 1.0, 0.3])
	X_obs_list = [np.hstack((X_obs, X_obs)) for i in range(output_dim)]
	Y_obs_list = [Y_obs[:, None]]

	# GMR prediction
	mu_gmr = []
	sigma_gmr = []
	H_gmr = []
	for i in range(Xt.shape[0]):
		mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])
		mu_gmr.append(mu_gmr_tmp)
		sigma_gmr.append(sigma_gmr_tmp)
		H_gmr.append(H_tmp)

	mu_gmr = np.array(mu_gmr)
	sigma_gmr = np.array(sigma_gmr)

	# GPR
	likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" %j, variance=noise_var) for j in range(output_dim)]
	kernel_list = [GPy.kern.Matern52(1, variance=1., lengthscale=1.) for i in range(gmr_model.nb_states)]

	# Fix variance of kernels
	for kernel in kernel_list:
		kernel.variance.fix(1.)
		kernel.lengthscale.fix(lengthscale)

	# GPR model
	K = Gmr_based_kernel(gmr_model=gmr_model, kernel_list=kernel_list)
	mf = GmrMeanMapping(2*input_dim+1, 1, gmr_model)

	m_obs = GPCoregionalizedWithMeanRegression(X_obs_list, Y_obs_list, kernel=K, likelihoods_list=likelihoods_list, mean_function=mf)

	# GPR prediction
	mu_gp, sigma_gp = m_obs.predict(Xtest, full_cov=True, Y_metadata={'output_index': output_index})

	mu_gp_rshp = np.reshape(mu_gp, (output_dim, -1)).T

	sigma_gp_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
	for i in range(output_dim):
		for j in range(output_dim):
			sigma_gp_tmp[:, :, i * output_dim + j] = sigma_gp[i * nb_data_test:(i + 1) * nb_data_test, j * nb_data_test:(j + 1) * nb_data_test]
	sigma_gp_rshp = np.zeros((nb_data_test, output_dim, output_dim))
	for i in range(nb_data_test):
		sigma_gp_rshp[i] = np.reshape(sigma_gp_tmp[i, i, :], (output_dim, output_dim))


	# Plots
	plt.figure(figsize=(5, 4))
	# GMM
	plot_gmm(np.array(gmr_model.mu), np.array(gmr_model.sigma), alpha=0.4, color=[0.1, 0.34, 0.73], zorder=10)
	# GMR based GPR prediction
	miny = mu_gp_rshp[:, 0] - np.sqrt(sigma_gp_rshp[:, 0, 0])
	maxy = mu_gp_rshp[:, 0] + np.sqrt(sigma_gp_rshp[:, 0, 0])
	plt.fill_between(Xt[:, 0], miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.3)
	plt.plot(Xt[:, 0], mu_gp_rshp[:, 0], color=[0.83, 0.06, 0.06], linewidth=3)
	plt.scatter(X_obs, Y_obs, color=[0, 0, 0], zorder=60, s=100)
	plt.axis('equal')
	axes = plt.gca()
	axes.set_xlim([-0.8, 2.8])
	# axes.set_ylim([-0.8, 2.2])
	plt.locator_params(nbins=4)
	plt.xlabel('$t$', fontsize=25)
	plt.ylabel('$y$', fontsize=25)
	plt.tick_params(labelsize=15)
	plt.tight_layout()
	plt.savefig('figures/GMRbGP_toyex_%dobs_lambda%f_noise%f.png' % (nb_obs, lengthscale, noise_var))
	plt.show()