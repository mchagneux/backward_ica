import argparse
import jax 
import jax.numpy as jnp
from backward_ica import hmm, utils

state_dim, obs_dim = 1,2
seq_length = 20
num_seqs = 10000
jax.config.update('jax_enable_x64',True)
utils.enable_x64(True)
import matplotlib.pyplot as plt 
import math

args = argparse.Namespace()
args.default_transition_bias = 0.5 
args.default_prior_base_scale = math.sqrt(1)
args.default_transition_base_scale = math.sqrt(1)
args.default_emission_base_scale = math.sqrt(1)
args.parametrization = 'cov_chol'

utils.set_global_cov_mode(args)

p_star = hmm.LinearGaussianHMM(state_dim, obs_dim,
                        transition_matrix_conditionning='diagonal',
                        range_transition_map_params=(0.95,1),
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


plt.axhline(y=avg_likelihood, linestyle='dotted', label='true logl')


best_logls = []
theta_mles = []
p_mle = hmm.LinearGaussianHMM(state_dim, obs_dim, 'diagonal')
for nb, key in enumerate(jax.random.split(key_mle, 1)):
    theta_mle, avg_logls, best_optim = p_mle.fit_kalman_rmle(key, 
                                    obs_seqs, 
                                    'sgd', 
                                    1e-3, 
                                    num_seqs // 100, 
                                    20, 
                                    theta_star)
    theta_mles.append(theta_mle)
    best_logls.append(avg_logls[best_optim])

    plt.plot(avg_logls, label=f'{nb}')

plt.legend()
plt.show()

key, key_gen_test = jax.random.split(key, 2)

# theta_mle = theta_mles[jnp.argmax(jnp.array(best_logls))]
state_seq, obs_seq = p_star.sample_seq(key_gen_test, theta_star, 100)

means, covs = p_star.smooth_seq(obs_seq, theta_star)

# print(covs)
fig, (ax0, ax1) = plt.subplots(2,1)
utils.plot_relative_errors_1D(ax0, state_seq[:,0], means[:,0], covs[:,0,0])
means, covs = p_mle.smooth_seq(obs_seq, theta_mle)

utils.plot_relative_errors_1D(ax1, state_seq[:,0], means[:,0], covs[:,0,0])
plt.show()

print('Theta star')
print(theta_star)

print('Theta MLE')
print(theta_mle)
# print('Optimal smoothing MSE:', jnp.abs(state_seq - smoothing_theta_star[0]).sum())
# print('MLE smoothing MSE:', jnp.abs(state_seq - smoothing_theta_mle[0]).sum())

# plt.show()

# print(theta_mle)
# print(theta_star)


