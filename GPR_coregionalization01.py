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
from utils.gmr import plot_gmm

# GPR (LMC) on 2D trajectories with time as input
if __name__ == '__main__':
	# Load data
	letter = 'B'  # choose a letter in the alphabet
	datapath = './data/2Dletters/'
	data = loadmat(datapath + '%s.mat' % letter)
	demos = [d['pos'][0][0].T for d in data['demos'][0]]

	# Parameters
	nb_data = demos[0].shape[0]
	nb_data_sup = 30
	nb_samples = 5
	dt = 0.01
	input_dim = 1
	output_dim = 2

	# Create time data
	demos_t = [np.arange(demos[i].shape[0])[:, None] for i in range(nb_samples)]
	# Stack time and position data
	demos_tx = [np.hstack([demos_t[i] * dt, demos[i]]) for i in range(nb_samples)]

	# Stack demos
	demos_np = demos_tx[0]
	for i in range(1, nb_samples):
		demos_np = np.vstack([demos_np, demos_tx[i]])

	X = demos_np[:, 0][:, None]
	Y = demos_np[:, 1:]

	X_list = [X for i in range(output_dim)]
	Y_list = [Y[:, i][:, None] for i in range(output_dim)]

	# Test data
	Xt = dt * np.arange(nb_data + nb_data_sup)[:, None]
	nb_data_test = Xt.shape[0]
	Xtest, _, output_index = GPy.util.multioutput.build_XY([Xt for i in range(output_dim)])

	# Create coregionalisation model
	kernel = GPy.kern.Matern52(input_dim, variance=1., lengthscale=10.)
	K = kernel.prod(GPy.kern.Coregionalize(1, output_dim, active_dims=[input_dim], name='B'))
	m = GPy.models.GPCoregionalizedRegression(X_list=X_list, Y_list=Y_list)
	m.randomize()
	m.optimize('bfgs', max_iters=100, messages=True)

	# Prediction
	mu_gp, sigma_gp = m.predict(Xtest, full_cov=True, Y_metadata={'output_index': output_index})

	mu_gp_rshp = np.reshape(mu_gp, (output_dim, -1))

	sigma_gp_tmp = np.zeros((nb_data_test, nb_data_test, output_dim*output_dim))
	for i in range(output_dim):
		for j in range(output_dim):
			sigma_gp_tmp[:, :, i*output_dim+j] = sigma_gp[i*nb_data_test:(i+1)*nb_data_test, j*nb_data_test:(j+1)*nb_data_test]
	sigma_gp_rshp = np.zeros((nb_data_test, output_dim, output_dim))
	for i in range(nb_data_test):
		sigma_gp_rshp[i] = np.reshape(sigma_gp_tmp[i, i, :], (2, 2))

	# Plots
	plt.figure(figsize=(5, 5))
	for p in range(nb_samples):
		plt.plot(Y[p*nb_data:(p+1)*nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 1], color=[.7, .7, .7])
		plt.scatter(Y[p*nb_data, 0], Y[p*nb_data, 1], color=[.7, .7, .7], marker='X', s=80)
	plot_gmm(mu_gp_rshp.T, sigma_gp_rshp, alpha=0.05, color=[0.99, 0.76, 0.53])
	plt.plot(mu_gp_rshp[0, :], mu_gp_rshp[1, :], color=[0.99, 0.76, 0.53], linewidth=3)
	plt.scatter(mu_gp_rshp[0, 0], mu_gp_rshp[1, 0], color=[0.99, 0.76, 0.53], marker='X', s=80)
	axes = plt.gca()
	axes.set_xlim([-14., 14.])
	axes.set_ylim([-14., 14.])
	plt.xlabel('$y_1$', fontsize=30)
	plt.ylabel('$y_2$', fontsize=30)
	plt.locator_params(nbins=3)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig('figures/GP_B.png')

	plt.figure(figsize=(5, 4))
	for p in range(nb_samples):
		plt.plot(Xt[:nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 0], color=[.7, .7, .7])
	plt.plot(Xt[:, 0], mu_gp_rshp[0, :], color=[0.99, 0.76, 0.53], linewidth=3)
	miny = mu_gp_rshp[0, :] - np.sqrt(sigma_gp_rshp[:, 0, 0])
	maxy = mu_gp_rshp[0, :] + np.sqrt(sigma_gp_rshp[:, 0, 0])
	plt.fill_between(Xt[:, 0], miny, maxy, color=[0.99, 0.76, 0.53], alpha=0.3)
	axes = plt.gca()
	axes.set_ylim([-14., 14.])
	plt.xlabel('$t$', fontsize=30)
	plt.ylabel('$y_1$', fontsize=30)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig('figures/GP_B01.png')

	plt.figure(figsize=(5, 4))
	for p in range(nb_samples):
		plt.plot(Xt[:nb_data, 0], Y[p*nb_data:(p+1)*nb_data, 1], color=[.7, .7, .7])
	plt.plot(Xt[:, 0], mu_gp_rshp[1, :], color=[0.99, 0.76, 0.53], linewidth=3)
	miny = mu_gp_rshp[1, :] - np.sqrt(sigma_gp_rshp[:, 1, 1])
	maxy = mu_gp_rshp[1, :] + np.sqrt(sigma_gp_rshp[:, 1, 1])
	plt.fill_between(Xt[:, 0], miny, maxy, color=[0.99, 0.76, 0.53], alpha=0.3)
	axes = plt.gca()
	axes.set_ylim([-14., 14.])
	plt.xlabel('$t$', fontsize=30)
	plt.ylabel('$y_2$', fontsize=30)
	plt.tick_params(labelsize=20)
	plt.tight_layout()
	plt.savefig('figures/GP_B02.png')
	plt.show()
