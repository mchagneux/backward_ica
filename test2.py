from backward_ica.hmm import LinearGaussianHMM
from backward_ica.kalman import Kalman, pykalman_logl_seq
from backward_ica.utils import set_global_cov_mode
import jax 
jax.config.update('jax_enable_x64', True)
d_z, d_x = 10,12 


import argparse 
args = argparse.Namespace()
import math
args.default_prior_base_scale = math.sqrt(1e-2)
args.default_transition_base_scale = math.sqrt(1e-2)
args.default_emission_base_scale = math.sqrt(1e-1)
args.default_transition_bias = 0.5
args.transition_bias = False
args.emission_bias = False
args.parametrization = 'cov_chol'

set_global_cov_mode(args)
p = LinearGaussianHMM(d_z, d_x, 'diagonal', False, False)

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key, 2)

theta = p.get_random_params(key)
state_seq, obs_seq = p.sample_seq(subkey, theta, 128)


logl = Kalman.filter_seq(obs_seq, p.format_params(theta))[-1]

logl_pykalman = pykalman_logl_seq(obs_seq, p.format_params(theta))

print(logl - logl_pykalman)