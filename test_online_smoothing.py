import argparse
import haiku as hk 
import jax 
import jax.numpy as jnp
import backward_ica.hmm as hmm 
import backward_ica.utils as utils 
import backward_ica.smc as smc
from backward_ica.svi import BackwardLinearTowerELBO, GeneralBackwardELBO, LinearGaussianTowerELBO, OnlineBackwardLinearTowerELBO
import seaborn as sns
import os 
import matplotlib.pyplot as plt

state_dim = 5
obs_dim = 5

args = argparse.Namespace()

args.parametrization = 'cov_chol'
import math
args.default_prior_mean = 0.0
args.range_transition_map_params = [0.99,1]
args.default_prior_base_scale = math.sqrt(1e-2)
args.default_transition_base_scale = math.sqrt(1e-2)
args.default_emission_base_scale = math.sqrt(1e-3)
args.default_transition_bias = 0
args.transition_bias = False
utils.set_global_cov_mode(args)

p = hmm.LinearGaussianHMM(state_dim, obs_dim, 'diagonal', (0.99,1), False, False)

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key, 2)

theta_star = p.get_random_params(subkey)

key, subkey = jax.random.split(key, 2)

state_seq, obs_seq = p.sample_seq(subkey, theta_star, 50)

num_samples = 1000
evidence = p.likelihood_seq(obs_seq, theta_star)
# normalizer = lambda x: jnp.mean(jnp.exp(x))
normalizer = smc.exp_and_normalize
closed_form_elbo = lambda obs_seq, theta, phi: LinearGaussianTowerELBO(p,p)(obs_seq, p.format_params(theta), p.format_params(phi))
offline_mc_elbo = lambda key, obs_seq, theta, phi: BackwardLinearTowerELBO(p, p, num_samples)(key, obs_seq, p.format_params(theta), p.format_params(phi))
online_mc_elbo = lambda key, obs_seq, theta, phi: OnlineBackwardLinearTowerELBO(p, p, normalizer, num_samples)(key, obs_seq, p.format_params(theta), p.format_params(phi))


true_elbo_value = closed_form_elbo(obs_seq, theta_star, theta_star)
offline_mc_elbo_value = offline_mc_elbo(key, obs_seq, theta_star, theta_star)

print('Closed-form ELBO error with evidence:', jnp.abs(true_elbo_value - evidence))
print('Offline Monte Carlo error:', jnp.abs(evidence - offline_mc_elbo_value))

online_mc_elbo_value = online_mc_elbo(key, obs_seq, theta_star, theta_star)
print('Online Monte Carlo error:', jnp.abs(evidence - online_mc_elbo_value))





