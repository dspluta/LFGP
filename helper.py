from scipy.io import loadmat
from scipy.linalg import expm, logm
import numpy as np


def load_data():
	data_odor = loadmat('/Users/linggeli/neuroscience/data/SuperChris/super_chris_extraction_odor2s.mat')
	trial_info = data_odor['trialInfo']
	select_odor_A = (trial_info[:, 0] == 1) & (trial_info[:, 1] == 1) & (trial_info[:, 3] == 1)
	lfp_odor_A = data_odor['lfpEpoch'][select_odor_A, 2000:4000, :]
	select_odor_B = (trial_info[:, 0] == 1) & (trial_info[:, 1] == 1) & (trial_info[:, 3] == 2)
	lfp_odor_B = data_odor['lfpEpoch'][select_odor_B, 2000:4000, :]
	return lfp_odor_A, lfp_odor_B


def log_euclidean_transform(lfp_data):
	cov_series = np.zeros((100, 5, 5))  # time series of covariance matrices
	utv_series = np.zeros((100, 15))  # time series of upper triangular vector
	for t in range(100):
		lfp_window = lfp_data[(t * 10):(t * 10 + 50), :]
		cov = np.cov(lfp_window, rowvar=False)
		cov_series[t, :, :] = cov
		utv = logm(cov)[np.triu_indices(5)]
		utv_series[t, :] = utv
	return utv_series
