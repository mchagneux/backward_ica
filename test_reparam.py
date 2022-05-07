from backward_ica.utils import GaussianParams, Scale
from backward_ica.hmm import LinearGaussianHMM
import jax.numpy as jnp 
import jax
import jax.random as random
from backward_ica.utils import *
d = 2
key = jax.random.PRNGKey(0)
# chol_true = jnp.tril(random.uniform(random.PRNGKey(0), (d,d)))
# print(chol_true)
# sigma_true = chol_true @ chol_true.T 
# scale = Scale(cov=sigma_true)
# print(scale.chol)

# # gaussian_params = GaussianParams(mean=jnp.ones((2,)), scale=Scale(chol=)


# cov = [[ 0.00659181, -0.00210874, -0.00260627],
#     [-0.00210874,  0.00805955, -0.0014159 ],
#     [-0.00260627, -0.0014159 ,  0.0055606 ]]

# mean = [ 0.40552006,  0.43618821,  0.38182363]

p = LinearGaussianHMM(2,2, 'diagonal')
theta = p.get_random_params(key)

state_seq, obs_seq = p.sample_seq(key, theta, 2)

theta = p.format_params(theta)
filt_states = p.compute_filt_state_seq(obs_seq, theta)
backwd_states = p.compute_kernel_state_seq(filt_states, theta)

keys = random.split(key, 3)

def sample_and_logpdf(key):
    sample = p.filt_dist.sample(key, tree_get_idx(-1, filt_states))
    return p.emission_kernel.logpdf(obs_seq[-1], sample, theta.emission)

print(vmap(sample_and_logpdf)(keys))





