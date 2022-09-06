import argparse
import jax 
import jax.numpy as jnp
from backward_ica import hmm, utils

state_dim, obs_dim = 3,4
seq_length = 10
num_seqs = 10000
jax.config.update('jax_enable_x64',True)
utils.enable_x64(True)
import matplotlib.pyplot as plt 
import math

args = argparse.Namespace()
args.default_prior_mean = 0.0
args.default_transition_bias = 0.5 
args.default_prior_base_scale = math.sqrt(0.1)
args.default_transition_base_scale = math.sqrt(0.1)
args.default_emission_base_scale = math.sqrt(0.1)
args.parametrization = 'cov_chol'

utils.set_defaults(args)

p_star = hmm.LinearGaussianHMM(state_dim, obs_dim,
                        transition_matrix_conditionning='diagonal',
                        range_transition_map_params=(0.99,1),
                        transition_bias=False,
                        emission_bias=False)



key = jax.random.PRNGKey(0)
key, key_params = jax.random.split(key, 2)
theta_star = p_star.get_random_params(key_params)
# print(p.format_params(theta))
key, key_gen = jax.random.split(key,2)
state_seqs, obs_seqs = p_star.sample_multiple_sequences(key_gen, theta_star, num_seqs, seq_length)

# plt.plot(obs_seqs[0])
# plt.show()


# my_likel = p.likelihood_seq(obs_seqs[0], theta)

# from backward_ica.kalman import pykalman_logl_seq

# print(my_likel - pykalman_logl_seq(obs_seqs[0], p.format_params(theta)))


key, key_mle = jax.random.split(key,2)

avg_likelihood = jax.vmap(p_star.likelihood_seq, in_axes=(0, None))(obs_seqs, theta_star).sum()


print('Logl', avg_likelihood)


plt.axhline(y=avg_likelihood, linestyle='dotted', label='$\log p_{\\theta}$')


best_logls = []
theta_mles = []
p_mle = hmm.LinearGaussianHMM(state_dim, obs_dim, 'diagonal')
for nb, key in enumerate(jax.random.split(key_mle, 3)):
    theta_mle, avg_logls, best_optim = p_mle.fit_kalman_rmle(key, 
                                    obs_seqs, 
                                    'sgd', 
                                    1e-3, 
                                    num_seqs // 100, 
                                    100, 
                                    theta_star)
    theta_mles.append(theta_mle)
    best_logls.append(avg_logls[best_optim])

    plt.plot(avg_logls, label=f'Fit {nb}')

plt.legend()
plt.savefig('train_log')
plt.clf()
key, key_gen_test = jax.random.split(key, 2)

theta_mle = theta_mles[jnp.argmax(jnp.array(best_logls))]
state_seq, obs_seq = p_star.sample_seq(key_gen_test, theta_star, 100)

means_star, covs_star = p_star.smooth_seq(obs_seq, theta_star)
means_mle, covs_mle = p_mle.smooth_seq(obs_seq, theta_mle)

# print(covs)
fig, axes = plt.subplots(state_dim,2, figsize=(15,15))
import numpy as np
axes = np.atleast_2d(axes)

for dim_nb in range(state_dim):
    
    utils.plot_relative_errors_1D(axes[dim_nb,0], state_seq[:,dim_nb], means_star[:,dim_nb], covs_star[:,dim_nb,dim_nb])
    utils.plot_relative_errors_1D(axes[dim_nb,1], state_seq[:,dim_nb], means_mle[:,dim_nb], covs_mle[:,dim_nb,dim_nb])

plt.autoscale(True)
plt.tight_layout()
plt.savefig('smoothing')

print('Theta star')
print(theta_star)

print('Theta MLE')
print(theta_mle)
# print('Optimal smoothing MSE:', jnp.abs(state_seq - smoothing_theta_star[0]).sum())
# print('MLE smoothing MSE:', jnp.abs(state_seq - smoothing_theta_mle[0]).sum())

# plt.show()

# print(theta_mle)
# print(theta_star)


