import haiku as hk 
import jax 
import jax.numpy as jnp
import backward_ica.hmm as hmm 
import backward_ica.utils as utils 
import backward_ica.smc as smc
from backward_ica.svi import GeneralBackwardELBO
import seaborn as sns
import os 
import matplotlib.pyplot as plt

state_dim = 1
obs_dim = 1

p = hmm.LinearGaussianHMM(state_dim, obs_dim, 'diagonal', (0.99,1), False, False)

key = jax.random.PRNGKey(0)

key, subkey = jax.random.split(key, 2)

theta_star = p.get_random_params(subkey)

key, subkey = jax.random.split(key, 2)

state_seq, obs_seq = p.sample_multiple_sequences(subkey, theta_star, 1, 50)


