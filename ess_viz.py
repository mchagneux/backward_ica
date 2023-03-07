#%%
import jax, jax.numpy as jnp
from backward_ica.stats.distributions import Gaussian, Scale
from backward_ica.stats.hmm import LinearGaussianHMM
from backward_ica.stats.kernels import Kernel
key = jax.random.PRNGKey(0)
from jax.flatten_util import ravel_pytree as ravel
num_samples = 4
mean = 0
params = Gaussian.Params(mean=jnp.array([0.]), scale=Scale(cov_chol=jnp.array([1.])))
from backward_ica.utils import tree_get_idx
import os 
import pandas as pd 
import matplotlib.pyplot as plt
from backward_ica.utils import exp_and_normalize, enable_x64
import dill
import seaborn as sns 
from scipy.stats import multivariate_normal
from functools import partial
enable_x64(True)
import numpy as np
import dill
path = 'experiments/p_chaotic_rnn/2023_03_06__15_34_34/neural_backward__online'
output_path = 'experiments/tests'
# online_elbo = pd.read_csv(os.path.join(path, 'neural_backward__online_tensorboard_logs_fit_0.csv'))['Value'].to_numpy()
# offline_elbo = pd.read_csv(os.path.join(path, 'neural_backward__online_tensorboard_logs_fit_0_monitor.csv'))['Value'].to_numpy()

# start_epoch = 150
# end_epoch = 300
# plt.plot(np.arange(start_epoch, end_epoch), online_elbo[start_epoch:end_epoch])
# plt.plot(np.arange(start_epoch, end_epoch), offline_elbo[start_epoch:end_epoch])


with open(os.path.join(path, 'weights'), 'rb') as f: 
    weights = dill.load(f)

log_q = jnp.array([weight[0] for weight in weights]).squeeze()
loq_q_backwd = jnp.array([weight[1] for weight in weights[1:]]).squeeze()
h_tilde = jnp.array([weight[2] for weight in weights[1:]]).squeeze()
ess = jnp.array([weight[3] for weight in weights]).squeeze()
ess = ess[:,1:,:]
# min_ess_sequence = jnp.min(ess, axis=(1,2))
mean_ess = jnp.mean(ess, axis=(1,2))
std_ess = jnp.std(ess, axis=(1,2))
# plt.plot(min_ess, label='Min ESS')
plt.plot(mean_ess, label='Mean ESS')
plt.plot(std_ess, label='Std ESS')
plt.xlabel('Epoch')
plt.legend()
# plt.plot(min_ess_sequence)

#%%
# index_largest_log_q = jnp.unravel_index(jnp.argmax(log_q), shape=log_q.shape)
# index_largest_log_q_backwd = jnp.unravel_index(jnp.argmax(loq_q_backwd), shape=loq_q_backwd.shape)
# index_smallest_log_q_backwd = jnp.unravel_index(jnp.argmin(loq_q_backwd), shape=loq_q_backwd.shape)

# index_smallest_log_q = jnp.unravel_index(jnp.argmin(log_q), shape=log_q.shape)

# print(jnp.array(index_largest_log_q))
# print(jnp.array(index_largest_log_q_backwd))
# print(jnp.array(index_smallest_log_q_backwd))

# index_max_weight = jnp.unravel_index(jnp.argmax(weights), weights.shape)
# index_min_weight = jnp.unravel_index(jnp.argmin(weights), weights.shape)



# print(np.array(index_min_weight))


#%%
# plt.plot(online_elbo, 'Online ELBO')
# plt.plot(offline_elbo, 'Offline ELBO')
# plt.savefig(os.path.join(output_path, 'training_curves.pdf'), format='pdf')
# # closed_form_curve = pd.read_csv('closed_form.csv')['Value']
# offline_csv = pd.read_csv('offline.csv')['Value']
# online_csv = pd.read_csv('online.csv')['Value']


# mask = jnp.array([False, True])
# print(test[mask])
# # # plt.plot(online_csv[2:])
# with open(path, 'rb') as f: 
#     results = dill.load(f)

# weights = jnp.concatenate([result[0] for result in results])
# samples = jnp.concatenate([result[1] for result in results])
# filt_params = [result[2] for result in results]
# backwd_params = [result[3] for result in results]
# filt_means = jnp.concatenate([filt_param.mean for filt_param in filt_params])
# filt_covs = jnp.concatenate([filt_param.scale.cov for filt_param in filt_params])
# backwd_ws = jnp.concatenate([backwd_param[0] for backwd_param in backwd_params])
# backwd_bs = jnp.concatenate([backwd_param[1] for backwd_param in backwd_params])
# backwd_covs = jnp.concatenate([backwd_param[2] for backwd_param in backwd_params])

# max_weight_position = jnp.unravel_index(jnp.argmax(weights), weights.shape)
# print(jnp.array(max_weight_position))

# E = max_weight_position[0]
# B = max_weight_position[1]
# t = max_weight_position[2]
# i = max_weight_position[3]
# weights_at_max = weights[E,B,t,i]

# ksi_t_i = samples[E,B,t,i]
# A, b, Sigma = backwd_ws[E,B,t], backwd_bs[E,B,t], backwd_covs[E,B,t]
# q_tm1_t_ksi_t_i_mean = A @ ksi_t_i + b 
# q_tm1_t_ksi_t_i_cov = Sigma
# q_tm1_params = (filt_means[E,B,t-1], filt_covs[E,B,t-1])
# q_tm1_t_ksi_t_pdf = lambda x,y: multivariate_normal.pdf(np.array([x,y]), mean=q_tm1_t_ksi_t_i_mean, cov=q_tm1_t_ksi_t_i_cov)
# q_tm1_pdf = lambda x,y: multivariate_normal.pdf(np.array([x,y]), mean=q_tm1_params[0], cov=q_tm1_params[1])
# x,y = np.meshgrid(np.linspace(-10,10,100), np.linspace(-10,10,100))
# z = jax.vmap(q_tm1_t_ksi_t_pdf, in_axes=(0,None))(x,y)
# q_tm1_pdf(x,y)
# plt.contour(x,y, )
# a = jax.vmap(q_tm1_pdf)(

# print(q_tm1_params[0])
# print(q_tm1_params[1])
# print(q_tm1_t_ksi_t_i_mean)
# print(q_tm1_t_ksi_t_i_cov)


#%%

# normalized_weights_where_min = exp_and_normalize(weights[402, 2, 6])

# print(normalized_weights_where_max)
# print(normalized_weights_where_min)
# max_weight_position = jnp.unravel_index(jnp.argmax(weights), weights.shape)
# min_weight_position = jnp.unravel_index(jnp.argmin(weights), weights.shape)
# normalized_weights_where_max = exp_and_normalize(weights[max_weight_position[:-1]])
# normalized_weights_where_min = exp_and_normalize(weights[min_weight_position[:-1]])

# print(normalized_weights_where_max)
# print(normalized_weights_where_min)

# print(jnp.array(max_weight_position))
# print(jnp.array(min_weight_position))

# print(jnp.exp(weights[min_weight_position]), jnp.array(min_weight_position))
# normalized_weights = jax.vmap(exp_and_normalize, in_axes=())(weights)
# print(weights.shape)

#%%
# plt.plot(closed_form_curve)

# test = 0
# path = 'experiments/p_linear/2023_01_19__19_23_36/linear__online_mc'
# weights = jnp.exp(jnp.load(os.path.join(path, 'weights.npy')))

# weights = weights.squeeze()
# max_weights = jnp.max(weights[:500], axis=(1,2,3))
# min_weights = jnp.min(weights[:500], axis=(1,2,3))

# # print(jnp.argmax(max_weights))
# plt.plot(max_weights)
# plt.savefig('max')
# plt.close()

# plt.plot(min_weights)
# plt.savefig('min')

# print(jnp.argmin(min_weights))
# max_weights = jnp.argmax(max_weights[jnp.argmax(max_weights)], axis=(1,2))

# plt.plot(max_weights)
# plt.savefig('max_weights.pdf', type='pdf')
# plt.close()
# plt.plot(min_weights)
# plt.savefig('min_weights.pdf', type='pdf')

# print(jnp.argmin(min_weights))
# plt.plot(min_weights)
#%%
# weight_max_values = jnp.max(problematic_weight,axis=(1,2))
# weight_min_values = jnp.min(problematic_weight,axis=(1,2))
# plt.plot(weight_max_values)
# plt.savefig('max_values')
# plt.close()
# # plt.plot(weight_min_values)s
# plt.savefig('min_values')
# weights_max_values = jnp.max(weights, axis=(1,2,3))
# print(jnp.argmax(weights_max_values))
# weights_min_values = jnp.min(weights, axis=(1,2,3))
# print(jnp.argmin(weights_min_values))

# plt.plot(weights_max_values, label='max_values')
# plt.legend()
# plt.savefig('max_values', format='pdf')
# plt.close()

# plt.plot(weights_min_values, label='min_values')
# plt.legend()
# plt.savefig('min_values', format='pdf')
# plt.close()

#%%