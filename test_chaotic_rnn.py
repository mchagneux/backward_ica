import argparse
from backward_ica.hmm import NonLinearHMM, HMM
from backward_ica.utils import set_parametrization
from jax import random



args = argparse.Namespace()

import math
args.parametrization = 'cov_chol' 
args.default_prior_mean = 0.0
args.default_prior_base_scale = math.sqrt(1e-2)
args.default_transition_base_scale = math.sqrt(1e-2)
args.default_emission_base_scale = math.sqrt(1e-2)
args.default_transition_bias = 0
args.default_emission_df = 2
args.default_emission_matrix = 1

args.state_dim, args.obs_dim = 2,3
args.grid_size = 0.001
args.gamma = 2.5
args.tau = 0.025
args.emission_matrix_conditionning = 'diagonal'
args.emission_bias = False
args.range_emission_map_params = (0.99,1)
args.num_particles = 1000
args.num_smooth_particles = 1000

args.injective = True
args.emission_map_layers = (8,)
args.slope = 0
args.transition_matrix_conditionning = 'diagonal'
args.range_transition_map_params = (0.99,1)

p = NonLinearHMM.chaotic_rnn(args)

theta = p.get_random_params(random.PRNGKey(0), args)
# formatted_theta = p.format_params(theta)
print(theta)
# state_seq, obs_seq = p.sample_seq(random.PRNGKey(0), theta, 100)