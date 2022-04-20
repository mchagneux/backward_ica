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
num_seqs = 4096 # number of sequences in the training set

batch_size = 64 # here a batch is a group of sequences 
learning_rate = 1e-2
num_epochs = 200
num_batches_per_epoch = num_seqs // batch_size
optimizer = optax.adam
num_samples = 1 # number of samples for the monte carlo approximation of the expectation of the (possibly nonlinear) emission term
num_particles = 1000
key = jax.random.PRNGKey(seed_model_params)
infer_key = jax.random.PRNGKey(seed_infer)

p = hmm.NonLinearGaussianHMM(state_dim=state_dim, 
                        obs_dim=obs_dim, 
                        transition_matrix_conditionning='diagonal') # specify the structure of the true model
                        
key, subkey = jax.random.split(key, 2)
theta = p.get_random_params(subkey) # sample params randomly (but covariances are fixed to default values)


with open('nonlinear_true_model.pickle','wb') as f:
    pickle.dump(theta, f)


key, subkey = jax.random.split(key, 2)
state_seqs, obs_seqs = hmm.sample_multiple_sequences(subkey, p.sample_seq, theta, num_seqs, seq_length)




#%% 
# smc_keys = jax.random.split(key, num_seqs)

# avg_evidence = jnp.mean(jax.vmap(jax.jit(lambda obs_seq, key: p.likelihood_seq(obs_seq, 
#                                                                     theta, 
#                                                                     key,
#                                                                     num_particles)))(obs_seqs, smc_keys))

# fast_smc_evidence = jax.jit(lambda obs_seq, key:smc_filter_seq(key, 
#                                                             obs_seq, 
#                                                             p.format_params(theta), 
#                                                             p.prior_sampler, 
#                                                             p.transition_kernel, 
#                                                             p.emission_kernel, 
#                                                             num_particles)[-1])

# avg_evidence_smc = jnp.mean(jax.vmap(fast_smc_evidence)(obs_seqs, smc_keys))
# print('Avg evidence:', avg_evidence)
# print('Avg evidence smc', avg_evidence_smc)
# filt_seq = smc_compute_filt_seq(key, obs_seqs[0], p.format_params(theta), p.prior_sampler, p.transition_kernel, p.emission_kernel, num_particles)
# smoothed_means, smoothed_covs = smc_smooth_from_filt_seq(key, filt_seq, p.format_params(theta), p.transition_kernel)

# # smoothed_means_kalman, smoothed_covs_kalman = p.smooth_seq(obs_seqs[0], theta)
import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1,1)
# utils.plot_relative_errors_1D(ax, state_seqs[0], smoothed_means, smoothed_covs)
# # utils.plot_relative_errors_1D(ax1, state_seqs[0], smoothed_means_kalman, smoothed_covs_kalman)

# plt.show()

# # phi, training_curves = p.multi_fit(key, obs_seqs, optimizer(learning_rate), batch_size, num_epochs, num_fits)
# # utils.plot_training_curves(training_curves, avg_evidence)
# import matplotlib.pyplot as plt

# q = hmm.LinearGaussianHMM(state_dim, obs_dim, None)

# trainer = SVITrainer(p, q, optimizer, learning_rate, num_epochs, batch_size, num_samples)


# phi, training_curves = trainer.multi_fit(infer_key, obs_seqs, theta, num_fits) # returns the best fit (based on the last value of the elbo)
# with open('linear_approx.pickle','wb') as f:
#     pickle.dump(phi, f)
    
# utils.plot_training_curves(training_curves, avg_evidence=avg_evidence)
# plt.savefig('training_curves_linearVI')
# plt.clf()

# utils.plot_example_smoothed_states(p, q, theta, phi, state_seqs, obs_seqs, 0, key, num_particles)
# plt.savefig('example_smoothing_linearVI')
# plt.clf()

# utils.plot_smoothing_wrt_seq_length_nonlinear(p, q, theta, phi, 256, 16, f'FFBSi_{num_particles}', 'linearVI', key, num_particles)
# plt.savefig('smoothing_perf_linearVI')
# plt.clf()



q = hmm.NeuralBackwardSmoother(state_dim=state_dim, 
                        obs_dim=obs_dim) # specify the structure of the true model, but init params are sampled during optimisiation     

trainer = SVITrainer(p, q, optimizer, learning_rate, num_epochs, batch_size, num_samples)


phi, training_curves = trainer.multi_fit(infer_key, obs_seqs, theta, num_fits) # returns the best fit (based on the last value of the elbo)
with open('nonlinear_approx.pickle','wb') as f:
    pickle.dump(phi, f)
    
# utils.plot_training_curves(training_curves, avg_evidence=avg_evidence)
# plt.savefig('training_curves_nonlinearVI')
# plt.clf()

utils.plot_example_smoothed_states(p, q, theta, phi, state_seqs, obs_seqs, 0, key, num_particles)
plt.savefig('example_smoothing_nonlinearVI')
plt.clf()

utils.plot_smoothing_wrt_seq_length_nonlinear(p, q, theta, phi, 256, 16, f'FFBSi_{num_particles}', 'nonlinearVI', key, num_particles)
plt.savefig('smoothing_perf_nonlinearVI')
plt.clf()



# num_particles = 1000
# prior_keys = jax.random.split(prior_key, num_particles)
# resampling_keys = jax.random.split(resampling_key, seq_length - 1)
# proposal_keys = jax.random.split(proposal_key, seq_length - 1)
# backwd_proposal_keys = jax.random.split(backwd_proposal_key, num_particles)

# smc_params = prior_keys, resampling_keys, proposal_keys, backwd_proposal_keys, num_particles
# utils.plot_example_smoothed_states(p, q, theta, phi, state_seqs, obs_seqs, 0, *smc_params)

# # utils.plot_smoothing_wrt_seq_length_nonlinear(key, 
#                                     p, 
#                                     q, 
#                                     theta, 
#                                     phi, 
#                                     seq_length=2048, 
#                                     step=64, 
#                                     ref_smoother_name = 'Kalman', 
#                                     approx_smoother_name = 'VI')
