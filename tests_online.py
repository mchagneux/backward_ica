import jax 
from jax import numpy as jnp
import seaborn as sns 
from backward_ica.stats.hmm import LinearGaussianHMM
from backward_ica.utils import * 
from datetime import datetime 
import os 
from backward_ica.elbos import GeneralBackwardELBO, OnlineGeneralBackwardELBO, LinearGaussianELBO
import pandas as pd 
import matplotlib.pyplot as plt

enable_x64(True)

date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
output_dir = os.path.join('experiments','online', date)
os.makedirs(output_dir, exist_ok=True)

key = jax.random.PRNGKey(1)
num_seqs = 200
seq_length = 50

state_dim = 1
obs_dim = 1
transition_matrix_conditionning = 'diagonal'
range_transition_map_params = (0,1)
transition_bias = True 
emission_bias = True 


num_samples = 100
normalizer = None

p = LinearGaussianHMM(state_dim, obs_dim, transition_matrix_conditionning, range_transition_map_params, transition_bias, emission_bias)
q = LinearGaussianHMM(state_dim, obs_dim, transition_matrix_conditionning, range_transition_map_params, transition_bias, emission_bias)

key, key_theta, key_phi = jax.random.split(key, 3)

theta = p.get_random_params(key_theta)
phi = q.get_random_params(key_phi)
phi = theta
true_elbo = jax.vmap(lambda obs_seq: LinearGaussianELBO(p, q)(obs_seq, len(obs_seq)-1, p.format_params(theta), p.format_params(phi)))
offline_elbo = jax.vmap(lambda key, obs_seq: GeneralBackwardELBO(p, q, num_samples)(key, obs_seq, len(obs_seq)-1, p.format_params(theta), p.format_params(phi)))
online_elbo = jax.vmap(lambda key, obs_seq: OnlineGeneralBackwardELBO(p,q, normalizer, num_samples).batch_compute(key, 
                                                                                                                obs_seq, 
                                                                                                                p.format_params(theta), 
                                                                                                                p.format_params(phi)))

true_eblo = jax.jit(true_elbo)
offline_elbo = jax.jit(offline_elbo)
online_elbo = jax.jit(online_elbo)

key, key_seqs = jax.random.split(key, 2)
state_seqs, obs_seqs = p.sample_multiple_sequences(key_seqs, theta, num_seqs, seq_length)


key, *elbo_keys = jax.random.split(key, num_seqs + 1)
elbo_keys = jnp.array(elbo_keys)
true_elbo_values = true_elbo(obs_seqs)
offline_elbo_values = offline_elbo(elbo_keys, obs_seqs)
online_elbo_values = online_elbo(elbo_keys, obs_seqs)

offline_errors = pd.DataFrame(true_elbo_values - offline_elbo_values)
online_errors = pd.DataFrame(true_elbo_values - online_elbo_values)

errors = pd.concat([offline_errors, online_errors], 
                    axis=1)
errors.columns = ['Offline','Online']
errors = errors.unstack().reset_index()
errors.columns = ['Method', 'Sequence','Values']

sns.boxplot(data=errors, x='Method', y='Values')
plt.savefig(os.path.join(output_dir, 'errors'))
plt.close()

sns.boxplot(data=offline_errors)
plt.savefig(os.path.join(output_dir, 'offline_errors'))
plt.close()

sns.boxplot(data=online_errors)
plt.savefig(os.path.join(output_dir, 'online_errors'))
plt.close()



# sns.boxplot(online_elbo_values)
# plt.savefig(os.path.join(output_dir, 'online_errors'))


