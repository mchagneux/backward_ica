#%% Imports

import jax 
import jax.numpy as jnp
import optax 
from jax import config 
config.update("jax_enable_x64", True)
from functools import partial

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVI, check_linear_gaussian_elbo
from IPython.display import display, Markdown


#%% Hyperparameters 
experiment_name = 'q_backward'
seed_model_params = 1326
seed_infer = 4569

num_fits = 2
state_dim, obs_dim = 1,2
seq_length = 128
num_seqs = 1024

batch_size = 64
learning_rate = 1e-2
num_epochs = 30
num_batches_per_epoch = num_seqs // batch_size
optimizer = optax.adam(learning_rate=learning_rate)

key = jax.random.PRNGKey(seed_model_params)
infer_key = jax.random.PRNGKey(seed_infer)

#%% Define p 
p = hmm.LinearGaussianHMM(state_dim=state_dim, 
                        obs_dim=obs_dim, 
                        transition_matrix_conditioning='diagonal')
key, subkey = jax.random.split(key, 2)
p_params = p.get_random_params(subkey)


#%% Define q 
q = hmm.LinearGaussianHMM(state_dim=state_dim, 
                        obs_dim=obs_dim, 
                        transition_matrix_conditioning='diagonal')

#%% Sample from p 
key, *subkeys = jax.random.split(key, num_seqs+1)
state_seqs, obs_seqs = jax.vmap(p.sample_seq, in_axes=(0, None, None))(jnp.array(subkeys), p_params, seq_length)

#%% 
# Sanity check available for linear gaussian case
check_linear_gaussian_elbo(obs_seqs, p, p_params)

# Evidence available via Kalman for linear gaussian case
evidence_seq = jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, p_params))(obs_seqs)
avg_evidence = jnp.mean(evidence_seq)

#%% Fit q
svi = SVI(p, q, optimizer, num_epochs, batch_size)

q_params, avg_elbos = svi.multi_fit(obs_seqs, p_params, infer_key, num_fits=num_fits)



#%% Plotting results 

utils.plot_fit_results_1D(p, q, p_params, q_params, state_seqs, obs_seqs, avg_elbos, avg_evidence)
utils.compare_mse_for_different_lengths(p, q, p_params, q_params, state_seqs, obs_seqs)
#%%