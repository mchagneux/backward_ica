import jax 
import jax.numpy as jnp
import optax 
from jax import config 
from tqdm import tqdm
import pickle

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer

seed_model_params = 1326 # one seed for the true model
seed_infer = 4569 # one seed for the approximating models 

num_fits = 5 # number of starting points for the optimisation of variational models
state_dim, obs_dim = 1,1
seq_length = 64 # length of each training sequence
num_seqs = 6400 # number of sequences in the training set

batch_size = 64 # here a batch is a group of sequences 
learning_rate = 1e-2
num_epochs = 200
optimizer = optax.adam
num_samples = 2 # number of samples for the monte carlo approximation of the expectation of the (possibly nonlinear) emission term
num_particles = 500

key = jax.random.PRNGKey(seed_model_params)
infer_key = jax.random.PRNGKey(seed_infer)

p = hmm.NonLinearGaussianHMM(state_dim=state_dim, 
                        obs_dim=obs_dim, 
                        transition_matrix_conditionning='diagonal') # specify the structure of the true model
                        
key, subkey = jax.random.split(key, 2)
theta = p.get_random_params(subkey) # sample params randomly (but covariances are fixed to default values)


with open('experiments/nonlinear_true_model.pickle','wb') as f:
    pickle.dump(theta, f)


key, subkey = jax.random.split(key, 2)
state_seqs, obs_seqs = hmm.sample_multiple_sequences(subkey, p.sample_seq, theta, num_seqs, seq_length)


#%% 
smc_keys = jax.random.split(key, num_seqs)

avg_evidence = jnp.mean(jax.vmap(jax.jit(lambda obs_seq, key: p.likelihood_seq(obs_seq, 
                                                                    theta, 
                                                                    key,
                                                                    num_particles)))(obs_seqs, smc_keys))


print('Avg evidence:', avg_evidence)


test_seq_length = 128
test_step = 16
timesteps = range(2,test_seq_length,test_step)
test_num_seqs = 2
test_state_seqs, test_obs_seqs = hmm.sample_multiple_sequences(key, p.sample_seq, theta, test_num_seqs, test_seq_length)

ffbsi_results = utils.multiple_length_ffbsi_smoothing(test_obs_seqs, p, theta, timesteps, key, num_particles)


q = hmm.LinearGaussianHMM(state_dim, obs_dim, None)

trainer = SVITrainer(p, q, optimizer, learning_rate, num_epochs, batch_size, num_samples)


phi, training_curves = trainer.multi_fit(infer_key, obs_seqs, theta, num_fits) # returns the best fit (based on the last value of the elbo)
with open('experiments/linear_VI.pickle','wb') as f:
    pickle.dump(phi, f)
    

utils.plot_training_curves(training_curves, figname='training_curves_linearVI', avg_evidence=avg_evidence)

approx_results = utils.multiple_length_linear_backward_smoothing(test_obs_seqs, q, phi, timesteps)
utils.plot_multiple_length_smoothing(test_state_seqs, ffbsi_results, approx_results, timesteps, 'ffbsi', 'VI', figname='ffbsi_vs_linearVI')


q = hmm.NeuralBackwardSmoother(state_dim=state_dim, 
                        obs_dim=obs_dim) # specify the structure of the true model, but init params are sampled during optimisiation     

trainer = SVITrainer(p, q, optimizer, learning_rate, num_epochs, batch_size, num_samples)


phi, training_curves = trainer.multi_fit(infer_key, obs_seqs, theta, num_fits) # returns the best fit (based on the last value of the elbo)
with open('experiments/nonlinear_VI.pickle','wb') as f:
    pickle.dump(phi, f)
    
utils.plot_training_curves(training_curves, figname='training_curves_nonlinearVI', avg_evidence=avg_evidence)
approx_results = utils.multiple_length_linear_backward_smoothing(test_obs_seqs, q, phi, timesteps)
utils.plot_multiple_length_smoothing(test_state_seqs, ffbsi_results, approx_results, timesteps, 'ffbsi', 'VI',figname='ffbsi_vs_nonlinearVI')


