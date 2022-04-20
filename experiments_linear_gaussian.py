import jax 
import jax.numpy as jnp
import optax 
from jax import config 
from tqdm import tqdm
import pickle

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer, check_linear_gaussian_elbo

seed_model_params = 1326 # one seed for the true model
seed_infer = 4569 # one seed for the approximating models 

num_fits = 2 # number of starting points for the optimisation of variational models
state_dim, obs_dim = 1,1
seq_length = 64 # length of each training sequence
num_seqs = 4096 # number of sequences in the training set

batch_size = 64 # here a batch is a group of sequences 
learning_rate = 1e-2
num_epochs = 80
num_batches_per_epoch = num_seqs // batch_size
optimizer = optax.adam

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
avg_evidence = jnp.mean(jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, theta))(obs_seqs))
print('Avg evidence:', avg_evidence)

# phi, training_curves = p.multi_fit(key, obs_seqs, optimizer(learning_rate), batch_size, num_epochs, num_fits)
# utils.plot_training_curves(training_curves, avg_evidence)


q = hmm.LinearGaussianHMM(state_dim=state_dim, 
                        obs_dim=obs_dim, 
                        transition_matrix_conditionning=None) # specify the structure of the true model, but init params are sampled during optimisiation     

trainer = SVITrainer(p, q, optimizer, learning_rate, num_epochs, batch_size)



phi, training_curves = trainer.multi_fit(infer_key, obs_seqs, theta, num_fits) # returns the best fit (based on the last value of the elbo)
utils.plot_training_curves(training_curves, avg_evidence=avg_evidence)


utils.plot_example_smoothed_states(p, p, theta, theta, state_seqs, obs_seqs, 0)

utils.plot_smoothing_wrt_seq_length_linear(key, 
                                    p, 
                                    q, 
                                    theta, 
                                    phi, 
                                    seq_length=2048, 
                                    step=32, 
                                    ref_smoother_name = 'Kalman', 
                                    approx_smoother_name = 'VI')
