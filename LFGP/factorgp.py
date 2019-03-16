import numpy as np
from mvn import build_covariance_blocks, sample_conditional_F
from utils import kernel_covariance


class FactorGP:
    """
    Latent factor Gaussian process model for multivariate time series
    Data: n epochs, t time points, q dimensional, r latent factors
    Parameters: loading matrix (r x q), variance vector (q), length scale (r)
    Priors: Conjugate Normal, inverse-Gamma, and Gamma (needs to be informative)
    """

    def __init__(self, dims, mu_var=[0, 1], inverse_gamma=[1, 1], gamma=[10, 1], F=None):
        n, t, q, r = dims
        self.dims = dims
        self.x = np.linspace(1, t, t)  # time points are indexed by intergers from 1 to t
        self.loading_prior_params = mu_var  # prior mean and variance for loading coeffcients
        self.variance_prior_params = inverse_gamma  # inverse Gamma prior for variance
        self.length_prior_params = gamma  # Gamma prior on length scale
        # self.kernel_type = 'default'
        self.loading, self.variance, self.theta = self.__initiate_params(dims, mu_var, inverse_gamma, gamma)
        self.F = F

    def __initiate_params(self, dims, mu_var, inverse_gamma, gamma):
        n, t, q, r = dims
        loading = np.random.normal(mu_var[0], np.sqrt(mu_var[1]), [r, q])
        variance = np.random.normal(0, 0.5, q) ** 2  # TODO: implement inverse-Gamma prior
        theta = np.repeat(gamma[0] * gamma[1], r)  # set length scale to gamma mean
        return loading, variance, theta

    def update_conditional_latent(self, Y):
        n, t, q, r = self.dims
        covs = []
        for l in self.theta:
            covs.append(kernel_covariance(self.x, l, 1.0))
        blocks = build_covariance_blocks(covs, self.loading, self.variance)
        F = np.zeros((n * t, r))
        for i in range(n):  # sample from F conditional distribution for each epoch independently
            F[(i * t):(i * t + t), :] = sample_conditional_F(Y[(i * t):(i * t + t), :], blocks)
        self.F = F

    def predict(self):
        return np.matmul(self.F, self.loading)