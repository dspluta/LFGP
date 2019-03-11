from model import *
from helper import *
import pickle


lfp_odor_A, lfp_odor_B = load_data()

for i in range(10):  # loop through 10 epochs for odor A
	lfp_data = lfp_odor_A[i, :, 5:10]
	utv_series = log_euclidean_transform(lfp_data)
	Y = utv_series - np.mean(utv_series, axis=0)
	results = run_model_sampler(Y, 2, 10)
	with open('results_odor_A_epoch_{}.pkl'.format(i + 1), 'wb') as fp:
		pickle.dump(results, fp)

for i in range(10):  # loop through 10 epochs for odor B
	lfp_data = lfp_odor_B[i, :, 5:10]
	utv_series = log_euclidean_transform(lfp_data)
	Y = utv_series - np.mean(utv_series, axis=0)
	results = run_model_sampler(Y, 2, 10)
	with open('results_odor_B_epoch_{}.pkl'.format(i + 1), 'wb') as fp:
		pickle.dump(results, fp)
