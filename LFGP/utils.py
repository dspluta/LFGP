import numpy as np
from scipy.stats import gamma, cauchy


def l_gamma_prior(l, a, b):
    """
    Gamma prior on length scale
    """
    return gamma.pdf(l, a=a, scale=1/b)


def s_half_cauchy_prior(s, scale):
    return 2 * cauchy.pdf(s, loc=0, scale=scale)


def kernel_covariance(x, l, s, noise=1e-6):
    """
    Covariance matrix with squared exponential kernel
    """
    t = x.shape[0]
    cov_sample = np.zeros((t, t))
    for i in range(t):
        for j in range(t):
            cov_sample[i, j] = s ** 2 * np.exp(-(x[i] - x[j]) ** 2 / (2 * l ** 2))
    cov_sample += np.eye(t) * noise  # add noise for numerical stability
    return cov_sample


def reshape_latent_curves(F, n, t):
    """
    Turn latent factors F of shape [nt, r] into a list of r factors of shape [t, n]
    """
    r = F.shape[1]
    F_curves_list = []
    for j in range(r):
        F_curves = np.zeros((t, n))
        for i in range(n):
            F_curves[:, i] = F[(i * t):(i * t + t), j]
        F_curves_list.append(F_curves)
    return F_curves_list
