from audioop import add
from time import time
import jax 
import jax.numpy as jnp
import optax 
from jax import config 
from tqdm import tqdm
import matplotlib.pyplot as plt

from backward_ica.kalman import pykalman_logl_seq

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer, check_linear_gaussian_elbo

seed_model_params = 1326 # one seed for the true model
seed_infer = 4569 # one seed for the approximating models 

num_fits = 5 # number of starting points for the optimisation of variational models
state_dim, obs_dim = 1,1
seq_length = 64 # length of each training sequence
num_seqs = 4096 # number of sequences in the training set

batch_size = 64 # here a batch is a group of sequences 
learning_rate = 1e-2
num_epochs = 150
num_batches_per_epoch = num_seqs // batch_size
optimizer = optax.adam
num_samples = 1 # number of samples for the monte carlo approximation of the expectation of the (possibly nonlinear) emission term

key = jax.random.PRNGKey(seed_model_params)
infer_key = jax.random.PRNGKey(seed_infer)

p = hmm.LinearGaussianHMM(state_dim=state_dim, 
                        obs_dim=obs_dim, 
                        transition_matrix_conditionning='diagonal') # specify the structure of the true model
                        
key, subkey = jax.random.split(key, 2)
theta = p.get_random_params(subkey) # sample params randomly (but covariances are fixed to default values)

key, subkey = jax.random.split(key, 2)
state_seqs, obs_seqs = hmm.sample_multiple_sequences(subkey, p.sample_seq, theta, num_seqs, seq_length)


check_linear_gaussian_elbo(obs_seqs, p, theta)

# fitted_params = p.multi_fit(key, obs_seqs, optimizer(learning_rate), batch_size, num_epochs)


q = hmm.LinearGaussianHMM(state_dim=state_dim, 
                        obs_dim=obs_dim, 
                        transition_matrix_conditionning='diagonal') # specify the structure of the true model, but init params are sampled during optimisiation     

trainer = SVITrainer(p, q, optimizer, learning_rate, num_epochs, batch_size, num_samples)

phi = trainer.multi_fit(infer_key, obs_seqs, theta, num_fits) # returns the best fit (based on the last value of the elbo)

utils.plot_example_smoothed_states(p, q, theta, phi, state_seqs, obs_seqs, 0)

utils.additive_smoothing_wrt_seq_length(key, p, q, theta, phi, seq_length=2048, step=64, reference_smoother_name = 'Kalman', approx_smoother_name = 'VI')
