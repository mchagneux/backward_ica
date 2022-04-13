
#%%
import jax 
import jax.numpy as jnp
import optax 
from jax import config 
config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.smc import smc_filter_seq, smc_smooth_seq
from backward_ica.svi import SVITrainer

seed_model_params = 1326 # one seed for the true model
seed_infer = 4569 # one seed for the approximating models 

num_fits = 5 # number of starting points for the optimisation of variational models
state_dim, obs_dim = 1,1
seq_length = 64 # length of each training sequence
num_seqs = 2048 # number of sequences in the training set

batch_size = 64 # here a batch is a group of sequences 
learning_rate = 1e-2 
num_epochs = 150 
num_batches_per_epoch = num_seqs // batch_size
optimizer = optax.adam(learning_rate=learning_rate)
num_samples = 1 # number of samples for the monte carlo approximation of the expectation of the (possibly nonlinear) emission term

# we get the same random keys therefore the transition kernel is identical to the one in the linear case
key = jax.random.PRNGKey(seed_model_params)
infer_key = jax.random.PRNGKey(seed_infer)
num_particles = 100 #number of particles for the smc runs

p = hmm.LinearGaussianHMM(state_dim=state_dim, 
                        obs_dim=obs_dim, 
                        transition_matrix_conditionning='diagonal') # the emission map is a fully connected neural network with 8 neurons in the hidden layer

key, subkey = jax.random.split(key, 2)
p_params = p.get_random_params(subkey)

key, subkey = jax.random.split(key, 2)
state_seqs, obs_seqs = hmm.sample_multiple_sequences(subkey, p.sample_seq, p_params, num_seqs, seq_length)



#%% 
prior_key, resampling_key, proposal_key = jax.random.split(key, 3)
prior_keys = jax.random.split(prior_key, num_seqs * num_particles).reshape(num_seqs, num_particles, -1)
resampling_keys = jax.random.split(resampling_key, num_seqs * (seq_length - 1)).reshape(num_seqs, seq_length - 1, -1)
proposal_keys = jax.random.split(proposal_key, num_seqs * (seq_length - 1)).reshape(num_seqs, seq_length - 1, -1)

#%%

likel_kalman = lambda obs_seq: p.likelihood_seq(obs_seq, p_params)

likel_smc = lambda prior_keys, resampling_keys, proposal_keys, obs_seq: smc_filter_seq(prior_keys, 
                                                                                    resampling_keys, 
                                                                                    proposal_keys, 
                                                                                    obs_seq, 
                                                                                    p.prior_sampler, 
                                                                                    p.transition_kernel, 
                                                                                    p.emission_kernel, 
                                                                                    p.format_params(p_params), 
                                                                                    num_particles)[-1]

# avg_evidence_smc = jnp.mean(jax.vmap(smc_likel)(obs_seqs, prior_keys, resampling_keys, proposal_keys)) # we compute the filtering distribution 
print('Avg evidence given by Kalman:', jnp.mean(jax.vmap(likel_kalman)(obs_seqs)))
print('Avg evidence given by SMC', jnp.mean(jax.vmap(likel_smc)(prior_keys, resampling_keys, proposal_keys, obs_seqs)))
#%%

tau = smc_smooth_seq(prior_keys[0], 
                    resampling_keys[0], 
                    proposal_keys[0], 
                    obs_seqs[0], 
                    p.prior_sampler, 
                    p.transition_kernel, 
                    p.emission_kernel, 
                    p.format_params(p_params), 
                    num_particles)

print('Smoothing with FFBSm',tau)
print('Smoothing with Kalman',jnp.sum(p.smooth_seq(obs_seqs[0], p_params)[0]))
#%%
q = hmm.LinearGaussianHMM(state_dim, obs_dim, 'diagonal')

trainer = SVITrainer(p, q, optimizer, num_epochs, batch_size, num_samples)

q_params, avg_elbos = trainer.multi_fit(obs_seqs, p_params, infer_key, num_fits=num_fits)

utils.plot_fit_results_1D(q, q_params, state_seqs, obs_seqs, avg_elbos, avg_evidence_smc)


#%%
q = hmm.NeuralSmoother(state_dim=state_dim, 
                        obs_dim=obs_dim) # this is a combination of three fully connected neural networks to update the backward and filtering distributions, as well a prior parameter and a shared parameter

trainer = SVITrainer(p, q, optimizer, num_epochs, batch_size, num_samples)

q_params, avg_elbos = trainer.multi_fit(obs_seqs, p_params, infer_key, num_fits=num_fits)

utils.plot_fit_results_1D(q, q_params, state_seqs, obs_seqs, avg_elbos, avg_evidence_smc)
# %%
