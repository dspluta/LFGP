import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

def se_kernel(theta, n_time):
    cov = np.empty([n_time, n_time])
    for s in range(n_time):
        for t in range(n_time):
            cov[s, t] = np.exp(-(s / n_time - t / n_time)**2 / theta**2)
    return(cov)

def simulate_LFGP(n_epochs, n_time, q, r, loading, v, kernel, theta, lowess_frac = 0.1):
    F = np.empty([n_epochs, n_time, r])
    Y = np.empty([n_epochs, n_time, q])
    x = np.linspace(0, 1, n_time)
    for i in range(n_epochs):
        for j in range(r):
            cov_F = kernel(theta[j, :], n_time)
            F[i, :, j] = lowess(np.random.multivariate_normal(np.repeat(0, n_time), cov_F, 1)[0],
                                x, is_sorted = True, frac = lowess_frac, it = 0)[:, 1]
        eps = np.random.multivariate_normal(np.repeat(0, q), np.diag(v), n_time)
        Y[i, :, :] = np.matmul(F[i, :, :], loading) + eps
    return({'Y': Y, 'F': F, 'loading': loading, 'v': v, 
            'kernel': kernel, 'theta': theta, 
            'lowess_frac': lowess_frac})
