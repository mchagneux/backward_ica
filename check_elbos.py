from argparse import ArgumentParser
from jax import numpy as jnp, random
from backward_ica.hmm import LinearGaussianHMM, NeuralLinearBackwardSmoother, NonLinearHMM
from backward_ica.svi import check_general_elbo, check_linear_gaussian_elbo, check_backward_linear_elbo
from backward_ica.kalman import Kalman, pykalman_logl_seq, pykalman_smooth_seq
from backward_ica.utils import GaussianParams, Scale, set_defaults
import matplotlib.pyplot as plt 
from backward_ica.utils import enable_x64
enable_x64(True)

key = random.PRNGKey(0)

state_dim, obs_dim = 10,12
num_seqs = 16
seq_length = 4
num_samples = 100
import argparse
import math
args = argparse.Namespace()

args.default_prior_base_scale = math.sqrt(1e-1)
args.default_transition_base_scale = math.sqrt(1e-2)
args.default_emission_base_scale = math.sqrt(1e-2)
args.default_transition_bias = -0.4
args.parametrization = 'cov_chol'
set_defaults(args)

p = LinearGaussianHMM(state_dim, obs_dim, 'diagonal', True, False)
# theta = p .get_random_params(key)

# state_seq, obs_seq  = p.sample_seq(key, theta, 16)

# marginals = p.smooth_seq(obs_seq, theta) 
# smoothed_means, smoothed_covs = Kalman.smooth_seq(obs_seq, p.format_params(theta))

# test = 0 
# check_linear_gaussian_elbo(p, num_seqs, seq_length)


# check_linear_gaussian_elbo(p, num_seqs, seq_length)

mc_key = random.PRNGKey(1)
check_linear_gaussian_elbo(p, num_seqs, seq_length)
check_backward_linear_elbo(mc_key, p, num_seqs, seq_length, num_samples)
check_general_elbo(mc_key, p, num_seqs, seq_length, num_samples)

# mc_key = random.PRNGKey(2)
# check_backward_linear_elbo(mc_key, p, num_seqs, seq_length, num_samples)
# check_general_elbo(mc_key, p, num_seqs, seq_length, num_samples)
# p = NonLinearHMM(state_dim, obs_dim, 'diagonal', (), 0, 10)

# key, subkey = random.split(key,2)

# theta = p.get_random_params(subkey)


# state_seq, obs_seq = p.sample_seq(subkey, theta, seq_length)




# # smoothed_means, smoothed_covs = p.smooth_seq(key, obs_seq, theta)

# # fig, ax = plt.subplots(1,1)
# # plot_relative_errors_1D(ax, state_seq, smoothed_means, smoothed_covs)
# # plt.savefig('smoothed')

# q = NeuralBackwardSmoother(state_dim, obs_dim, update_layers=(8,), backwd_layers=(8,))

# phi = q.format_params(q.get_random_params(key))


# filt_params = q.init_filt_params(obs_seq[0], phi)

# print('First filt state:', filt_params)
# backwd_params = q.new_backwd_params(filt_params, phi)
# # print('First backwd state:', backwd_params)

# filt_params = q.new_filt_params(obs_seq[1], filt_params, phi)
# print('Second filt state:', filt_params)


# print('Output of first backwd kernel map:', q.backwd_kernel.map(state_seq[1], backwd_params))
# print('State to be mapped:', state_seq[1])
# print(q.backwd_kernel.sample(key, state_seq[1], backwd_params))



# smoothed_means, smoothed_covs = p.smooth_seq(obs_seq, theta) 
# smoothed_means_2, smoothed_covs_2 = Kalman.smooth_seq(obs_seq, p.format_params(theta))
# smoothed_means_3, smoothed_covs_3 = pykalman_smooth_seq(obs_seq, p.format_params(theta))
# test = 0 


# check_linear_gaussian_elbo(p, num_seqs, seq_length)
# check_general_elbo(p, num_seqs, seq_length, num_samples)



# import haiku as hk 

# def gru(obs, state):
#     net = hk.GRU(8)
#     return net(obs, state)

# lstm = hk.without_apply_rng(hk.transform(gru))


# dummy_obs = jnp.ones((2,))
# dummy_state = jnp.ones((8,))

# init_params = lstm.init(random.PRNGKey(0), dummy_obs, dummy_state)

# out, new_state = lstm.apply(init_params, dummy_obs, dummy_state)
# test = 0 


# import backward_ica.utils as utils 

# theta_good = utils.load_params('theta', 'experiments/q_phi_linear_gaussian')
# theta_weird = utils.load_params('theta', 'experiments/linear_model_5')

# print(theta_good)
# print(theta_weird)

# import matplotlib.pyplot as plt
# points = jnp.linspace(-2,2,100)
# plt.plot(points, jnp.tanh(points))
# plt.show()








