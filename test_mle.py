import argparse
import jax 
from backward_ica import hmm, utils

state_dim, obs_dim = 10,10
seq_length = 100
num_seqs = 1000
jax.config.update('jax_enable_x64',True)


args = argparse.Namespace()
args.default_transition_bias = 0.5 
args.default_prior_base_scale = 1e-2
args.default_transition_base_scale = 1e-2
args.default_emission_base_scale = 1e-2
args.parametrization = 'cov_chol'
utils.set_global_cov_mode(args)

p = hmm.LinearGaussianHMM(state_dim, obs_dim,
                        transition_matrix_conditionning='diagonal',
                        transition_bias=False,
                        emission_bias=False)
key = jax.random.PRNGKey(0)
key, key_params = jax.random.split(key, 2)
theta = p.get_random_params(key_params)
print(theta)
# print(p.format_params(theta))
key, key_gen = jax.random.split(key,2)
state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta, num_seqs, seq_length)
key, key_mle = jax.random.split(key,2)

avg_likelihood = jax.vmap(p.likelihood_seq, in_axes=(0, None))(obs_seqs, theta).mean()
print(avg_likelihood)
theta_mle, avg_logls = p.fit_kalman_rmle(key_mle, obs_seqs, 'adam', 1e-1, num_seqs // 20, 10)

import matplotlib.pyplot as plt 

plt.plot(avg_logls)
plt.axhline(y=avg_likelihood, linestyle='dotted')
plt.show()