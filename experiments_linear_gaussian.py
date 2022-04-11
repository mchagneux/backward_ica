#%% Imports

import jax 
import jax.numpy as jnp
import optax 
from jax import config 
config.update("jax_enable_x64", True)
from functools import partial

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer, check_linear_gaussian_elbo
from IPython.display import display, Markdown
from argparse import Namespace



#%% Hyperparameters 
experiment_name = 'q_backward'
seed_model_params = 1326
seed_infer = 4569

num_fits = 5
state_dim, obs_dim = 1,2
seq_length = 64
num_seqs = 1024

batch_size = 64
learning_rate = 1e-2
num_epochs = 50
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
sampler = jax.vmap(p.sample_seq, in_axes=(0, None, None))
state_seqs, obs_seqs = sampler(jnp.array(subkeys), p_params, seq_length)

#%% 
# Sanity check available for linear gaussian case
check_linear_gaussian_elbo(obs_seqs, p, p_params)

# Evidence available via Kalman for linear gaussian case
evidence_seq = jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, p_params))(obs_seqs)
avg_evidence = jnp.mean(evidence_seq)

#%% Fit q
trainer = SVITrainer(p, q, optimizer, num_epochs, batch_size)

q_params, avg_elbos = trainer.multi_fit(obs_seqs, p_params, infer_key, num_fits=num_fits)

#%% Plotting results 
utils.plot_fit_results_1D(p, q, p_params, q_params, state_seqs, obs_seqs, avg_elbos, avg_evidence)

num_test_seqs = 32
test_seqs_length = 4096
test_state_seqs, test_obs_seqs = sampler(jax.random.split(key, num_test_seqs), p_params, test_seqs_length)

utils.compare_mse_for_different_lengths(p, q, p_params, q_params, test_state_seqs, test_obs_seqs, step=64)
#%%