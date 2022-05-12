from backward_ica.utils import GaussianParams, Scale
from backward_ica.hmm import LinearGaussianHMM
import jax.numpy as jnp 
import jax
import jax.random as random
from backward_ica.utils import *
d = 2
enable_x64(True)
key = jax.random.PRNGKey(0)
# chol_true = jnp.tril(random.uniform(random.PRNGKey(0), (d,d), minval=10, maxval=20))
# # print(chol_true)
# sigma_true = chol_true @ chol_true.T 
# scale = Scale(prec=sigma_true)
# scale2 = Scale(cov=scale.cov)

# print(scale.cov)
# print(scale.prec)
# print(scale.prec_chol)
# print(scale.cov_chol)
# print('----')
# print(scale2.cov)
# print(scale2.prec)
# print(scale2.prec_chol)
# print(scale2.cov_chol)
# print(scale.prec)


# # gaussian_params = GaussianParams(mean=jnp.ones((2,)), scale=Scale(chol=)


# cov = [[ 0.00659181, -0.00210874, -0.00260627],
#     [-0.00210874,  0.00805955, -0.0014159 ],
#     [-0.00260627, -0.0014159 ,  0.0055606 ]]

# mean = [ 0.40552006,  0.43618821,  0.38182363]

p = LinearGaussianHMM(2,2, 'diagonal')
theta = p.get_random_params(key)
# formatted_theta = p.format_params(theta)
# print(formatted_theta)
# theta = p.format_params(theta)

# print(theta.transition.scale.prec)
# print(theta.emission.scale.prec)

state_seq, obs_seq = p.sample_seq(key, theta, 16)

# theta = p.format_params(theta)
# filt_states = p.compute_filt_state_seq(obs_seq, theta)
# backwd_states = p.compute_kernel_state_seq(filt_states, theta)

# keys = random.split(key, 3)

# def sample_and_logpdf(key):
#     sample = p.filt_dist.sample(key, tree_get_idx(-1, filt_states))
#     return p.emission_kernel.logpdf(obs_seq[-1], sample, theta.emission)

# print(vmap(sample_and_logpdf)(keys))





