#%%

import jax 
import jax.numpy as jnp
import optax 
from jax import config 
import matplotlib.pyplot as plt
config.update("jax_enable_x64", True)


import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVI, check_linear_gaussian_elbo

#%% Hyperparameters 
experiment_name = 'q_backward'
seed_model_params = 1326
seed_infer = 4569

num_starting_points = 10
state_dim, obs_dim = 1,2
seq_length = 64
num_seqs = 2048

batch_size = 8
learning_rate = 1e-3
num_epochs = 100
num_batches_per_epoch = num_seqs // batch_size
optimizer = optax.adam(learning_rate=learning_rate)

key = jax.random.PRNGKey(seed_model_params)
infer_key = jax.random.PRNGKey(seed_infer)

p = hmm.LinearGaussianHMM(state_dim=state_dim, obs_dim=obs_dim, transition_matrix_conditioning='diagonal')
key, subkey = jax.random.split(key, 2)
p_params = p.get_random_params(subkey)

key, *subkeys = jax.random.split(key, num_seqs+1)
state_seqs, obs_seqs = jax.vmap(p.sample_seq, in_axes=(0, None, None))(jnp.array(subkeys), p_params, seq_length)

check_linear_gaussian_elbo(obs_seqs, p, p_params)

#%%

evidence_seq = jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, p_params))(obs_seqs)


#%% Define q 
q = hmm.LinearGaussianHMM(state_dim=state_dim, obs_dim=obs_dim, transition_matrix_conditioning='diagonal')

#%% Fit q

svi = SVI(p, q, optimizer, num_epochs, batch_size)
avg_evidence = jnp.mean(evidence_seq)

for sub_exp_nb, infer_subkey in enumerate(jax.random.split(infer_key, num_starting_points)):
    q_params = q.get_random_params(infer_subkey)
    q_params, avg_elbos = svi.fit(obs_seqs, p_params, q_params)

    fig = plt.figure(figsize=(10,10))

    ax0 = fig.add_subplot(131)
    ax0.plot(avg_elbos, label='$\mathcal{L}(\\theta,\\phi)$')
    ax0.axhline(y=avg_evidence, c='red', label = '$log p_{\\theta}(x)$' )
    ax0.set_xlabel('Epoch') 
    ax0.set_title(f'{num_seqs} seqs of {seq_length} obs')

    seq_nb = 0
    ax1 = fig.add_subplot(132)
    utils.plot_relative_errors_1D(ax1, state_seqs[seq_nb], *p.smooth_seq(obs_seqs[seq_nb], p_params))
    ax1.set_title('Kalman')

    ax2 = fig.add_subplot(133, sharey=ax1)
    utils.plot_relative_errors_1D(ax2, state_seqs[seq_nb], *q.smooth_seq(obs_seqs[seq_nb], q_params))
    ax2.set_title('Backward variational')

    plt.autoscale(True)
    plt.legend()
    plt.show()
#%%