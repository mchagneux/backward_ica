#%%
from backward_ica.elbo import LinearELBO, NonLinearELBO
import backward_ica.hmm as hmm
from backward_ica.kalman import filter as kalman_filter, smooth as kalman_smooth
import jax 
import jax.numpy as jnp
import haiku as hk 
import optax 
key = jax.random.PRNGKey(123)
key, subkey = jax.random.split(key,2)
state_dim, obs_dim = 1,3
seq_length = 2
num_seqs = 256
from jax import config 
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt


p_params, p_def = hmm.get_random_params(subkey, 
                                    state_dim, 
                                    obs_dim,
                                    transition_mapping_type='linear',
                                    emission_mapping_type='linear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_def)


key, *subkeys = jax.random.split(key, num_seqs+1)
state_seqs, obs_seqs = jax.vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), seq_length)

kalman_log_likel = lambda obs_seq: kalman_filter(obs_seq, p)[-1]
avg_evidence = jnp.mean(jax.vmap(kalman_log_likel)(obs_seqs))
print('Average evidence:',avg_evidence)

#%%

d = state_dim

def backwd(filt_mean, filt_cov):
    net = hk.nets.MLP((16, d**2 + d + d*(d+1) // 2))
    out = net(jnp.concatenate((filt_mean, jnp.tril(filt_cov).flatten())))
    A = out[:d**2].reshape((d,d))
    a = out[d**2:d**2+d]
    cov = jnp.zeros((d,d))
    cov = cov.at[jnp.tril_indices(d)].set(out[d**2+d:])
    return A, a, cov @ cov.T

def filt(obs, filt_mean, filt_cov):
    net = hk.nets.MLP((16, d + d*(d+1) // 2))
    out = net(jnp.concatenate((obs, filt_mean, jnp.tril(filt_cov).flatten())))
    mean = out[:d]
    cov = jnp.zeros((d,d))
    cov = cov.at[jnp.tril_indices(d)].set(out[d:])
    return mean, cov @ cov.T

key, *subkeys = jax.random.split(key, 3)

filt_init, filt_apply = hk.without_apply_rng(hk.transform(filt))
backward_init, backward_apply = hk.without_apply_rng(hk.transform(backwd))

dummy_obs = obs_seqs[0][0]
dummy_mean = jnp.empty((state_dim,))
dummy_cov = jnp.empty((state_dim, state_dim))

filt_params = filt_init(subkeys[0], dummy_obs, dummy_mean, dummy_cov)
backward_params = backward_init(subkeys[1], dummy_mean, dummy_cov)


q_def = {'filtering':filt_apply, 
        'backward':backward_apply}

q_params = {'filtering':filt_params,
            'backward':backward_params}
#%% 

elbo = NonLinearELBO(p_def, q_def).compute
loss = lambda obs_seq, key, q_params: elbo(obs_seq, key, p_params, q_params)

optimizer = optax.adam(learning_rate=1e-3)

batch_size = 8
num_batches_per_epoch = num_seqs // batch_size
     
@jax.jit
def q_step(q_params, opt_state, batch, keys):
    loss_values, grads = jax.vmap(jax.value_and_grad(loss, argnums=2), in_axes=(0,0,None))(batch, keys, q_params)
    avg_loss_value = jnp.mean(loss_values)
    avg_grads = jax.tree_util.tree_map(jnp.mean, grads)
    updates, opt_state = optimizer.update(avg_grads, opt_state, q_params)
    q_params = optax.apply_updates(q_params, updates)
    return q_params, opt_state, avg_loss_value

num_epochs = 1500
key, *subkeys = jax.random.split(key, num_seqs * num_epochs + 1)
subkeys = jnp.array(subkeys).reshape(num_epochs,num_seqs,-1)

def fit(q_params):
    opt_state = optimizer.init(q_params)
    avg_neg_elbos = []

    def loader(obs_seqs, keys):
        for index in range(0, seq_length, batch_size):
            yield obs_seqs[index:index + batch_size], keys[index:index + batch_size]

    for epoch_nb in range(num_epochs):
        avg_neg_elbo_epoch = 0.0
        for (batch, keys) in loader(obs_seqs, subkeys[epoch_nb]):
            q_params, opt_state, avg_neg_elbo = q_step(q_params, opt_state, batch, keys)
            avg_neg_elbo_epoch += avg_neg_elbo / num_batches_per_epoch
        avg_neg_elbos.append(avg_neg_elbo_epoch)

    return q_params, avg_neg_elbos

fitted_q_params, avg_neg_elbos = fit(q_params)
get_marginals = lambda obs_seq: NonLinearELBO(p_def, q_def).compute_tractable_terms(obs_seq, p, fitted_q_params)[1]
marginal_means, marginal_covs = jax.vmap(get_marginals)(obs_seqs)

print('Expectation under q_phi against true states:',jnp.mean((marginal_means - state_seqs)**2))

plt.plot(-jnp.array(avg_neg_elbos))
plt.xlabel('Epoch nb') 
plt.ylabel('$\mathcal{L}(\\theta,\\phi)$')
plt.tight_layout()
plt.autoscale(True)
plt.show()
#%%

smoothed_means_kalman, smoothed_covs_kalman = jax.vmap(kalman_smooth, in_axes=(0, None))(obs_seqs, p)

def plot_relative_errors(seq_nb):
    time_axis = range(seq_length)
    kalman_means = smoothed_means_kalman[seq_nb]
    variational_means = marginal_means[seq_nb]

    plt.scatter(time_axis, (kalman_means-variational_means) / kalman_means)

#%%
# import numpy as np
# from matplotlib.patches import Ellipse
# import matplotlib.transforms as transforms

# def visualize_predictions_on_sequence(states, predicted_means, predicted_covs):

#     for state, predicted_mean, predicted_cov in zip(states, predicted_means, predicted_covs): 
#         plt.scatter(state[0],state[1], c='r')
#         plt.scatter(predicted_mean[0], predicted_mean[1], c='tab:purple')
#         cov = predicted_cov
#         n_std = 2
#         pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

#         # Using a special case to obtain the eigenvalues of this
#         # two-dimensionl dataset.
#         ell_radius_x = np.sqrt(1 + pearson)
#         ell_radius_y = np.sqrt(1 - pearson)
#         ellipse = Ellipse((0,0),
#             width=ell_radius_x * 2,
#             height=ell_radius_y * 2, fill=False)

#         scale_x = np.sqrt(cov[0, 0]) * n_std
#         mean_x = predicted_mean[0]

#         # calculating the stdandard deviation of y ...
#         scale_y = np.sqrt(cov[1, 1]) * n_std
#         mean_y = predicted_mean[1]

#         transf = transforms.Affine2D() \
#             .rotate_deg(45) \
#             .scale(scale_x, scale_y) \
#             .translate(mean_x, mean_y)

#         ellipse.set_transform(transf + plt.gca().transData)

#         plt.gca().add_patch(ellipse)

#     plt.show()

# # %%
# visualize_predictions_on_sequence(states, smoothed_means_kalman, smoothed_covs_kalman)
# visualize_predictions_on_sequence(states, marginal_means, marginal_covs)
# # %% Compare with linear 