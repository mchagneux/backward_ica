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
num_seqs = 2048

batch_size = 64
learning_rate = 1e-2
num_epochs = 70
num_batches_per_epoch = num_seqs // batch_size
optimizer = optax.adam(learning_rate=learning_rate)
num_samples = 1

key = jax.random.PRNGKey(seed_model_params)
infer_key = jax.random.PRNGKey(seed_infer)

#%% Define p 
p = hmm.NonLinearGaussianHMM(state_dim=state_dim, 
                        obs_dim=obs_dim, 
                        transition_matrix_conditionning='diagonal')
key, subkey = jax.random.split(key, 2)
p_params = p.get_random_params(subkey)


#%% Define q 
q = hmm.NeuralSmoother(state_dim=state_dim, 
                        obs_dim=obs_dim)

#%% Sample from p 
key, *subkeys = jax.random.split(key, num_seqs+1)
sampler = jax.vmap(p.sample_seq, in_axes=(0, None, None))
state_seqs, obs_seqs = sampler(jnp.array(subkeys), p_params, seq_length)


#%% Fit q
trainer = SVITrainer(p, q, optimizer, num_epochs, batch_size, num_samples)

q_params, avg_elbos = trainer.multi_fit(obs_seqs, p_params, infer_key, num_fits=num_fits)

#%% Plotting results 
utils.plot_fit_results_1D(q, q_params, state_seqs, obs_seqs, avg_elbos)

num_test_seqs = 32
test_seqs_length = 4096
test_state_seqs, test_obs_seqs = sampler(jax.random.split(key, num_test_seqs), p_params, test_seqs_length)

utils.compare_mse_for_different_lengths(q, q_params, test_state_seqs, test_obs_seqs, step=64)
#%%