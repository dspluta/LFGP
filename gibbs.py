import matplotlib as mpl 
mpl.use('TkAgg')

import pymc3 as pm
import theano.tensor as tt

import numpy as np

from pymc3 import Metropolis


def sample_covariance(X, l, s):
	"""
	Create Squared-exponential covariance matrix for X given length scale l and variance scale s.
	"""
	t = X.shape[0]
	cov_sample = np.zeros((t, t))
	for i in range(t):
		for j in range(t):
			# TODO: the distance calculation is ad-hoc
			cov_sample[i, j] = s ** 2 * np.exp(-(i * 0.1 - j * 0.1) ** 2 / (2 * l ** 2))
	return cov_sample


def matern32(d, l):
	"""
	Matern function with nu=1.5
	"""
	return (1. + np.sqrt(3. * d ** 2) / l) * np.exp(-np.sqrt(3. * d ** 2) / l)


def sample_covariance_matern(X, l, s):
	"""
	Create Matern covariance matrix.
	"""
	t = X.shape[0]
	cov_sample = np.zeros((t, t))
	for i in range(t):
		for j in range(t):
			cov_sample[i, j] = s ** 2 * matern32(i * 0.1 - j * 0.1, l)
	return cov_sample


def sample_gp_posterior(F_column, n_sample=100, gprior_params=(10, 0.1),  prior_scale=0.5, sigma=1e-3, test=False):
	"""
	Sample from GP posterior given one column of F (single factor).
	
	Args
		F_column: (numpy array) [t] latent observations at t time points
		n_sample: (int) MCMC chain length
		gprior_params: (tuple) Gamma prior on length scale (distance between adjacent time points is always 0.1)
		prior_scale: (float) half-Cauchy prior scale
		sigma: (float) Gaussian process noise (set to small number for noise free latent process)
		test: (bool) test mode returns the trace
	"""
	t = F_column.shape[0]
	X = np.linspace(0.1, t * 0.1, t).reshape((t, 1))
	with pm.Model() as model:
		l = pm.Gamma('l', gprior_params[0], gprior_params[1])  # informative prior for length scale
		s = pm.HalfCauchy('s', prior_scale)  # recommended prior for scale parameter
		K = s ** 2 * pm.gp.cov.ExpQuad(1, l)  
		#K = s ** 2 * pm.gp.cov.Matern32(1, l)  # Matern (nu=1.5) kernel
		gp = pm.gp.Marginal(cov_func=K)
		y_ = gp.marginal_likelihood('y', X=X, y=F_column, noise=sigma)
		trace = pm.sample(n_sample, Metropolis(), chains=1)  # use Metropolis instead of NUTS for speed
		#trace = pm.sample(n_sample)
	l = trace['l'][-1]
	s = trace['s'][-1]
	cov = sample_covariance(X, l, s)
	#mu, cov = gp.predict(X, point=trace[-1])
	if test:
		return cov, trace
	else:
		return cov


def build_covariance_blocks(F_covariance_list, loading_matrix, Y_variance):
	"""
	Build covariance matrix for long vector of all columns of Y stacked together.
	
	Args
		F_covariance_list: (list) of [t, t] covariance matrices
		loading_matrix: (numpy array) [r, q] linear transformation between F and Y
		Y_sigma_list: (numpy array) [q] variance parameters for columns of Y
	"""
	r = len(F_covariance_list)
	t = F_covariance_list[0].shape[0]
	q = loading_matrix.shape[1]
	block_YY = np.zeros((q * t, q * t))
	# covariance for columns of F
	block_FF_rows = []
	for i in range(r):
		current_row = np.zeros((t, r * t))
		current_row[:, (i * t):(i * t + t)] = F_covariance_list[i]
		block_FF_rows.append(current_row)
	block_FF = np.vstack(block_FF_rows)
	# covariance between columns of F and columns of Y
	block_FY_rows = []
	for i in range(r):
		current_row = np.zeros((t, q * t))
		for j in range(q):
			current_row[:, (j * t):(j * t + t)] = loading_matrix[i, j] * F_covariance_list[i]
		block_FY_rows.append(current_row)
	block_FY = np.vstack(block_FY_rows)
	block_YF = np.transpose(block_FY)
	# covariance between columns of Y
	block_YY_rows = []
	for i in range(q):
		current_row = np.zeros((t, q * t))
		for j in range(q):
			for k in range(r):
				current_row[:, (j * t):(j * t + t)] += F_covariance_list[k] * loading_matrix[k, i] * loading_matrix[k, j]
			if i == j:
				current_row[:, (j * t):(j * t + t)] += np.eye(t) * Y_variance[i]  # diagonal variance
		block_YY_rows.append(current_row)
	block_YY = np.vstack(block_YY_rows)
	return block_FF, block_FY, block_YF, block_YY


def sample_conditional_F(Y, block_FF, block_FY, block_YF, block_YY, debug=False):
	"""
	Sample from conditional distribution of F given everything else.
	
	Args
		Y: (numpy array) [t, q] observed multivariate time series
		block_FF, block_FY, block_YF, block_YY: (numpy array) blocks in the covariance of joint distribution
	"""
	t, q = Y.shape
	r = int(block_FF.shape[0] / t)
	Y_stack = np.transpose(Y).reshape(t * q)  # stack columns of Y
	block_YY_inverse = np.linalg.inv(block_YY)
	prod = np.matmul(block_FY, block_YY_inverse)
	mu = np.matmul(prod, Y_stack)
	covariance = block_FF - np.matmul(prod, block_YF)
	F_stack = np.random.multivariate_normal(mu, covariance)
	F_sample = np.transpose(F_stack.reshape((r, t)))  # de-stack columns of F
	if debug:
		return F_sample, covariance
	else:
		return F_sample

		
def blr(y, F, mu_0, Sigma_0, a_0, b_0, n_draws=1):
	n = y.shape[0]
	p = mu_0.shape[0]
	mu_post = np.matmul(np.linalg.inv(np.matmul(np.transpose(F), F) + Sigma_0),
						np.matmul(Sigma_0, mu_0) + np.matmul(np.transpose(F), y))
	Sigma_post = np.matmul(np.transpose(F), F) + Sigma_0
	a_post = a_0 + n / 2
	b_post = b_0 + 0.5 * (np.matmul(np.transpose(y), y) + 
						  np.matmul(np.matmul(np.transpose(mu_0), Sigma_0), mu_0) - 
						  np.matmul(np.matmul(np.transpose(mu_post), Sigma_post), mu_post))
	beta = np.empty([n_draws, p])
	sigma2_eps = 1 / np.random.gamma(a_post, 1 / b_post, n_draws)
	
	for i in range(n_draws):
		beta[i, :] = np.random.multivariate_normal(mu_post, sigma2_eps[i] * np.linalg.inv(Sigma_post))
	return beta, sigma2_eps


def blr_mv(y, F, mu_0, Sigma_0, a_0, b_0):
	q = y.shape[1]
	r = F.shape[1]
	beta_est = np.empty([r, q])
	sigma2_eps_est = np.empty(q)
	for j in range(q):
		results = blr(y[:, j], F, mu_0, Sigma_0, a_0, b_0)
		beta_est[:, j] = results[0]
		sigma2_eps_est[j] = results[1]
	return beta_est, sigma2_eps_est
