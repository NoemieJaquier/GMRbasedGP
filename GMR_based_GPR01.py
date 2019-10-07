#! /usr/bin/env python
##################################################
# Author: Noemie Jaquier, 2019
# License: MIT
# Contact: noemie.jaquier@idiap.ch
##################################################

import numpy as np
import GPy
import matplotlib.pyplot as plt
from scipy.io import loadmat # loading data from matlab
from utils.gmr import Gmr
from utils.gmr import plot_gmm
from utils.gp_coregionalize_with_mean_regression import GPCoregionalizedWithMeanRegression
from utils.gmr_mean_mapping import GmrMeanMapping
from utils.gmr_kernels import Gmr_based_kernel


# GMR-based GPR on 2D trajectories with time as input
if __name__ == '__main__':
	letter = 'B'  # choose a letter in the alphabet
	datapath = './data/2Dletters/'
	data = loadmat(datapath + '%s.mat' % letter)
	demos = [d['pos'][0][0].T for d in data['demos'][0]]

	# Parameters
	nb_data = demos[0].shape[0]
	nb_data_sup = 50
	nb_samples = 5
	dt = 0.01
	input_dim = 1
	output_dim = 2
	in_idx = [0]
	out_idx = [1, 2]
	nb_states = 6

	nb_prior_samples = 10
	nb_posterior_samples = 3

	# Create time data
	demos_t = [np.arange(demos[i].shape[0])[:, None] + 1 for i in range(nb_samples)]
	# Stack time and position data
	demos_tx = [np.hstack([demos_t[i]*dt, demos[i]]) for i in range(nb_samples)]

	# Stack demos
	demos_np = demos_tx[0]
	for i in range(1, nb_samples):
		demos_np = np.vstack([demos_np, demos_tx[i]])

	X = demos_np[:, 0][:, None]
	Y = demos_np[:, 1:]

	# Train data for GPR
	X_list = [np.hstack((X, X)) for i in range(output_dim)]
	Y_list = [Y[:, i][:, None] for i in range(output_dim)]

	# Test data
	Xt = dt * np.arange(demos[0].shape[0] + nb_data_sup)[:, None]
	nb_data_test = Xt.shape[0]
	Xtest, _, output_index = GPy.util.multioutput.build_XY([np.hstack((Xt, Xt)) for i in range(output_dim)])

	# Define via-points (new set of observations)
	X_obs = np.array([0.0, 1., 1.9])[:, None]
	Y_obs = np.array([[-12.5, -11.5], [-0.5, -1.5], [-14.0, -7.5]])
	X_obs_list = [np.hstack((X_obs, X_obs)) for i in range(output_dim)]
	Y_obs_list = [Y_obs[:, i][:, None] for i in range(output_dim)]

	# GMM
	gmr_model = Gmr(nb_states=nb_states, nb_dim=input_dim + output_dim, in_idx=in_idx, out_idx=out_idx)
	gmr_model.init_params_kbins(demos_np.T, nb_samples=nb_samples)
	gmr_model.gmm_em(demos_np.T)

	# GMR prediction
	mu_gmr = []
	sigma_gmr = []
	for i in range(Xt.shape[0]):
		mu_gmr_tmp, sigma_gmr_tmp, H_tmp = gmr_model.gmr_predict(Xt[i])
		mu_gmr.append(mu_gmr_tmp)
		sigma_gmr.append(sigma_gmr_tmp)

	mu_gmr = np.array(mu_gmr)
	sigma_gmr = np.array(sigma_gmr)

	# Define GPR likelihood and kernels
	likelihoods_list = [GPy.likelihoods.Gaussian(name="Gaussian_noise_%s" %j, variance=0.01) for j in range(output_dim)]
	# kernel_list = [GPy.kern.RBF(1, variance=1., lengthscale=0.1) for i in range(gmr_model.nb_states)]
	kernel_list = [GPy.kern.Matern52(1, variance=1., lengthscale=5.) for i in range(gmr_model.nb_states)]

	# Fix variance of kernels
	for kernel in kernel_list:
		kernel.variance.fix(1.)
		kernel.lengthscale.constrain_bounded(0.01, 10.)

	# Bound noise parameters
	for likelihood in likelihoods_list:
		likelihood.variance.constrain_bounded(0.001, 0.05)

	# GPR model
	K = Gmr_based_kernel(gmr_model=gmr_model, kernel_list=kernel_list)
	mf = GmrMeanMapping(2*input_dim+1, 1, gmr_model)

	m = GPCoregionalizedWithMeanRegression(X_list, Y_list, kernel=K, likelihoods_list=likelihoods_list, mean_function=mf)

	# Parameters optimization
	m.optimize('bfgs', max_iters=100, messages=True)

	# Print model parameters
	print(m)

	# GPR prior (no observations)
	prior_traj = []
	prior_mean = mf.f(Xtest)[:, 0]
	prior_kernel = m.kern.K(Xtest)
	for i in range(nb_prior_samples):
		prior_traj_tmp = np.random.multivariate_normal(prior_mean, prior_kernel)
		prior_traj.append(np.reshape(prior_traj_tmp, (output_dim, -1)))

	prior_kernel_tmp = np.zeros((nb_data_test, nb_data_test, output_dim * output_dim))
	for i in range(output_dim):
		for j in range(output_dim):
			prior_kernel_tmp[:, :, i * output_dim + j] = prior_kernel[i * nb_data_test:(i + 1) * nb_data_test, j * nb_data_test:(j + 1) * nb_data_test]
	prior_kernel_rshp = np.zeros((nb_data_test, output_dim, output_dim))
	for i in range(nb_data_test):
		prior_kernel_rshp[i] = np.reshape(prior_kernel_tmp[i, i, :], (output_dim, output_dim))

	# GPR posterior -> new points observed (the training points are discarded as they are "included" in the GMM)
	m_obs = GPCoregionalizedWithMeanRegression(X_obs_list, Y_obs_list, kernel=K, likelihoods_list=likelihoods_list, mean_function=mf)
	mu_posterior_tmp = m_obs.posterior_samples_f(Xtest, full_cov=True, size=nb_posterior_samples)

	mu_posterior = []
	for i in range(nb_posterior_samples):
		mu_posterior.append(np.reshape(mu_posterior_tmp[:, 0, i], (output_dim, -1)))

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

	# Final plots
	# GMM
	plt.figure(figsize=(5, 5))
	for p in range(nb_samples):
		plt.plot(Y[p * nb_data:(p + 1) * nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
		plt.plot(Y[p * nb_data, 0], Y[p * nb_data, 1], color=[.7, .7, .7], marker='o')
	plot_gmm(np.array(gmr_model.mu)[:, 1:], np.array(gmr_model.sigma)[:, 1:, 1:], alpha=0.6, color=[0.1, 0.34, 0.73])
	plt.savefig('figures/GMRbGP_B_gmm.png')

	# Priors
	plt.figure(figsize=(5, 5))
	plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
	plt.scatter(mu_gmr[0, 0], mu_gmr[0, 1], color=[0.20, 0.54, 0.93], marker='X', s=80)
	plot_gmm(mu_gmr, prior_kernel_rshp, alpha=0.05, color=[0.64, 0.27, 0.73])
	for i in range(nb_prior_samples):
		plt.plot(prior_traj[i][0], prior_traj[i][1], color=[0.64, 0.27, 0.73], linewidth=1.)
		plt.scatter(prior_traj[i][0, 0], prior_traj[i][1, 0], color=[0.64, 0.27, 0.73], marker='X', s=80)
	axes = plt.gca()
	axes.set_xlim([-17., 17.])
	axes.set_ylim([-17., 17.])
	plt.xlabel('$y_1$', fontsize=30)
	plt.ylabel('$y_2$', fontsize=30)
	plt.locator_params(nbins=3)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig('figures/GMRbGP_B_priors_datasup.png')

	plt.figure(figsize=(5, 4))
	plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3)
	miny = mu_gmr[:, 0] - np.sqrt(prior_kernel_rshp[:, 0, 0])
	maxy = mu_gmr[:, 0] + np.sqrt(prior_kernel_rshp[:, 0, 0])
	plt.fill_between(Xt[:, 0], miny, maxy, color=[0.64, 0.27, 0.73], alpha=0.3)
	for i in range(nb_prior_samples):
		plt.plot(Xt[:, 0], prior_traj[i][0], color=[0.64, 0.27, 0.73], linewidth=1.)
	axes = plt.gca()
	axes.set_ylim([-17., 17.])
	plt.xlabel('$t$', fontsize=30)
	plt.ylabel('$y_1$', fontsize=30)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig('figures/GMRbGP_B_priors01_datasup.png')

	plt.figure(figsize=(5, 4))
	for p in range(nb_samples):
		plt.plot(Xt[:nb_data, 0], Y[p * nb_data:(p + 1) * nb_data, 1], color=[.7, .7, .7])
	plt.plot(Xt[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
	miny = mu_gmr[:, 1] - np.sqrt(prior_kernel_rshp[:, 1, 1])
	maxy = mu_gmr[:, 1] + np.sqrt(prior_kernel_rshp[:, 1, 1])
	plt.fill_between(Xt[:, 0], miny, maxy, color=[0.64, 0.27, 0.73], alpha=0.3)
	for i in range(nb_prior_samples):
		plt.plot(Xt[:, 0], prior_traj[i][1], color=[0.64, 0.27, 0.73], linewidth=1.)
	axes = plt.gca()
	axes.set_ylim([-17., 17.])
	plt.xlabel('$t$', fontsize=30)
	plt.ylabel('$y_2$', fontsize=30)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig('figures/GMRbGP_B_priors02_datasup.png')
	plt.show()

	# Posterior
	plt.figure(figsize=(5, 5))
	plt.plot(mu_gmr[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
	plot_gmm(mu_gp_rshp, sigma_gp_rshp, alpha=0.05, color=[0.83, 0.06, 0.06])
	for i in range(nb_posterior_samples):
		plt.plot(mu_posterior[i][0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)
		plt.scatter(mu_posterior[i][0, 0], mu_posterior[i][1, 0], color=[0.64, 0., 0.65], marker='X', s=80)
	plt.plot(mu_gp_rshp[:, 0], mu_gp_rshp[:, 1], color=[0.83, 0.06, 0.06], linewidth=3.)
	plt.scatter(mu_gp_rshp[0, 0], mu_gp_rshp[0, 1], color=[0.83, 0.06, 0.06], marker='X', s=80)
	plt.scatter(Y_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)
	axes = plt.gca()
	axes.set_xlim([-17., 17.])
	axes.set_ylim([-17., 17.])
	plt.xlabel('$y_1$', fontsize=30)
	plt.ylabel('$y_2$', fontsize=30)
	plt.locator_params(nbins=3)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig('figures/GMRbGP_B_posterior_datasup.png')

	plt.figure(figsize=(5, 4))
	plt.plot(Xt[:, 0], mu_gmr[:, 0], color=[0.20, 0.54, 0.93], linewidth=3.)
	miny = mu_gp_rshp[:, 0] - np.sqrt(sigma_gp_rshp[:, 0, 0])
	maxy = mu_gp_rshp[:, 0] + np.sqrt(sigma_gp_rshp[:, 0, 0])
	plt.fill_between(Xt[:, 0], miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.3)
	for i in range(nb_posterior_samples):
		plt.plot(Xt[:, 0], mu_posterior[i][0], color=[0.64, 0., 0.65], linewidth=1.5)
	plt.plot(Xt[:, 0], mu_gp_rshp[:, 0], color=[0.83, 0.06, 0.06], linewidth=3)
	plt.scatter(X_obs[:, 0], Y_obs[:, 0], color=[0, 0, 0], zorder=60, s=100)
	axes = plt.gca()
	axes.set_ylim([-17., 17.])
	plt.xlabel('$t$', fontsize=30)
	plt.ylabel('$y_1$', fontsize=30)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig('figures/GMRbGP_B_posterior01_datasup.png')

	plt.figure(figsize=(5, 4))
	plt.plot(Xt[:, 0], mu_gmr[:, 1], color=[0.20, 0.54, 0.93], linewidth=3.)
	miny = mu_gp_rshp[:, 1] - np.sqrt(sigma_gp_rshp[:, 1, 1])
	maxy = mu_gp_rshp[:, 1] + np.sqrt(sigma_gp_rshp[:, 1, 1])
	plt.fill_between(Xt[:, 0], miny, maxy, color=[0.83, 0.06, 0.06], alpha=0.3)
	for i in range(nb_posterior_samples):
		plt.plot(Xt[:, 0], mu_posterior[i][1], color=[0.64, 0., 0.65], linewidth=1.5)
	plt.plot(Xt[:, 0], mu_gp_rshp[:, 1], color=[0.83, 0.06, 0.06], linewidth=3)
	plt.scatter(X_obs[:, 0], Y_obs[:, 1], color=[0, 0, 0], zorder=60, s=100)
	axes = plt.gca()
	axes.set_ylim([-17., 17.])
	plt.xlabel('$t$', fontsize=30)
	plt.ylabel('$y_2$', fontsize=30)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig('figures/GMRbGP_B_posterior02_datasup.png')
	plt.show()

