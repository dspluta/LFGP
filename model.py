from gibbs import *

from sklearn.decomposition import PCA
from tqdm import tqdm as tqdm


def initiate_factors(Y, latent_dim):
	"""
	Initiate latent Gaussian process factors with principal components.
	"""
	t = Y.shape[0]
	pca = PCA(n_components=latent_dim)
	components = pca.fit_transform(Y)
	mu_0 = np.repeat(0, latent_dim)  # prior mean 0 for regression coefficients
	Sigma_0 = np.diag(np.repeat(1, latent_dim))  # prior variance 1
	a_0 = 1  # Inverse-Gamma(1, 1) is fairly diffused
	b_0 = 1
	loading_matrix, Y_variance = blr_mv(Y, components, mu_0, Sigma_0, a_0, b_0)
	X = np.linspace(0.1, t * 0.1, t).reshape((t, 1))  # create initial GP covariance matrices
	cov1 = sample_covariance_matern(X, 1.0, 1.0)  # length scale 1.0 corresponds to 10 time points
	cov2 = sample_covariance_matern(X, 1.0, 1.0)  # variance scale set to 1.0 as well
	S1, S2, S3, S4 = build_covariance_blocks([cov1, cov2], loading_matrix, Y_variance)
	F = sample_conditional_F(Y, S1, S2, S3, S4)
	return F


def gibbs_sampling(F, Y, verbose=True):
	""" 
	One Gibbs sampling step to update everything else given F then draw from conditional of F.
	"""
	latent_dim = F.shape[1]
	mu_0 = np.repeat(0, latent_dim)  # prior mean 0 for regression coefficients
	Sigma_0 = np.diag(np.repeat(1, latent_dim))  # prior variance 1
	a_0 = 1  # Inverse-Gamma(1, 1) is fairly diffused
	b_0 = 1
	loading_matrix, Y_variance = blr_mv(Y, F, mu_0, Sigma_0, a_0, b_0)
	Y_hat = np.matmul(F, loading_matrix)
	mse = np.mean((Y - Y_hat) ** 2)
	if verbose:
		print(mse)
	covs = []
	gp_traces = []
	for j in range(latent_dim):
		cov, gp_trace = sample_gp_posterior(F[:, j], test=True)
		covs.append(cov)
		gp_traces.append(gp_trace)
	S1, S2, S3, S4 = build_covariance_blocks(covs, loading_matrix, Y_variance)
	F = sample_conditional_F(Y, S1, S2, S3, S4)
	return F, loading_matrix, Y_variance, gp_traces, mse


def run_model_sampler(Y, latent_dim, n_iter):
	"""
	Create model and run Gibbs sampler for n iterations.
	"""
	F_sample = []
	loading_sample = []
	variance_sample = []
	trace_sample = []
	mse_history = []
	F = initiate_factors(Y, latent_dim)
	for i in tqdm(range(n_iter)):
		F, loading_matrix, Y_variance, gp_traces, mse = gibbs_sampling(F, Y)
		F_sample.append(F)
		loading_sample.append(loading_matrix)
		variance_sample.append(Y_variance)
		trace_sample.append(gp_traces)
		mse_history.append(mse)
	return F_sample, loading_sample, variance_sample, trace_sample, mse_history
