#%%
from backward_ica.elbo import LinearELBO, NonLinearELBO, QFromBackward, QFromForward
import backward_ica.hmm as hmm
from backward_ica.kalman import filter as kalman_filter, smooth as kalman_smooth
import jax 
import jax.numpy as jnp
import haiku as hk 
import optax 

#%% Define p 
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key,2)
state_dim, obs_dim = 1,2
seq_length = 32
num_seqs = 256
from jax import config 
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt


p_params, p_model = hmm.get_random_params(subkey, 
                                    state_dim, 
                                    obs_dim,
                                    transition_mapping_type='linear',
                                    emission_mapping_type='linear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_model)


key, *subkeys = jax.random.split(key, num_seqs+1)
state_seqs, obs_seqs = jax.vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), seq_length)

kalman_log_likel = lambda obs_seq: kalman_filter(obs_seq, p)[-1]
avg_evidence = jnp.mean(jax.vmap(kalman_log_likel)(obs_seqs))
print('Average log-evidence:',avg_evidence)
#%% Define q 

d = state_dim

def backwd(filt_mean, filt_cov):
    net = hk.nets.MLP((8,8, d**2 + d + d*(d+1) // 2))
    out = net(jnp.concatenate((filt_mean, jnp.tril(filt_cov).flatten())))
    A = out[:d**2].reshape((d,d))
    a = out[d**2:d**2+d]
    cov = jnp.zeros((d,d))
    cov = cov.at[jnp.tril_indices(d)].set(out[d**2+d:])
    return A, a, cov @ cov.T

def filt_predict(filt_mean, filt_cov):
    net = hk.nets.MLP((8,8, d + d*(d+1) // 2))
    out = net(jnp.concatenate((filt_mean, jnp.tril(filt_cov).flatten())))
    mean = out[:d]
    cov = jnp.zeros((d,d))
    cov = cov.at[jnp.tril_indices(d)].set(out[d:])
    return mean, cov @ cov.T

def filt_update(obs, pred_mean, pred_cov):
    net = hk.nets.MLP((8,8, d + d*(d+1) // 2))
    out = net(jnp.concatenate((obs, pred_mean, jnp.tril(pred_cov).flatten())))
    mean = out[:d]
    cov = jnp.zeros((d,d))
    cov = cov.at[jnp.tril_indices(d)].set(out[d:])
    return mean, cov @ cov.T

key, *subkeys = jax.random.split(key, 4)

filt_predict_init, filt_predict_apply = hk.without_apply_rng(hk.transform(filt_predict))
filt_update_init, filt_update_apply = hk.without_apply_rng(hk.transform(filt_update))
backward_init, backward_apply = hk.without_apply_rng(hk.transform(backwd))

dummy_obs = obs_seqs[0][0]
dummy_mean = jnp.empty((state_dim,))
dummy_cov = jnp.empty((state_dim, state_dim))

filt_predict_params = filt_predict_init(subkeys[0], dummy_mean, dummy_cov)
filt_update_params = filt_update_init(subkeys[1], dummy_obs, dummy_mean, dummy_cov)
backward_params = backward_init(subkeys[2], dummy_mean, dummy_cov)


q_model = {'filtering':{'predict':filt_predict_apply, 'update':filt_update_apply},
        'backward':backward_apply}

q_params = {'filtering':{'predict':filt_predict_params, 'update':filt_update_params},
            'backward':backward_params}

key, subkey = jax.random.split(key, 2)
q_params, q_model = hmm.get_random_params(subkey, state_dim, obs_dim, 'linear','linear')
elbo = NonLinearELBO(p_model, QFromForward(q_model)).compute
#%% Fit q

# elbo = NonLinearELBO(p_model, QFromBackward(q_model)).compute

loss = lambda obs_seq, key, q_params: elbo(obs_seq, key, p_params, q_params)

optimizer = optax.adam(learning_rate=1e-3)

batch_size = 8
num_batches_per_epoch = num_seqs // batch_size
     
@jax.jit
def q_step(q_params, opt_state, batch, keys):
    neg_elbo_values, grads = jax.vmap(jax.value_and_grad(loss, argnums=2), in_axes=(0,0,None))(batch, keys, q_params)
    summed_elbo_values = jnp.sum(-neg_elbo_values)
    avg_grads = jax.tree_util.tree_map(jnp.mean, grads)
    updates, opt_state = optimizer.update(avg_grads, opt_state, q_params)
    q_params = optax.apply_updates(q_params, updates)
    return q_params, opt_state, summed_elbo_values

num_epochs = 500
key, *subkeys = jax.random.split(key, num_seqs * num_epochs + 1)
subkeys = jnp.array(subkeys).reshape(num_epochs,num_seqs,-1)

def fit(q_params):
    opt_state = optimizer.init(q_params)
    avg_elbos = []

    def loader(obs_seqs, keys):
        for batch_start in range(0, num_seqs, batch_size):
            yield obs_seqs[batch_start:batch_start + batch_size], keys[batch_start:batch_start + batch_size]

    for epoch_nb in range(num_epochs):
        avg_elbo_epoch = 0.0
        for batch_nb, (batch, keys) in enumerate(loader(obs_seqs, subkeys[epoch_nb])):
            q_params, opt_state, summed_elbo_values = q_step(q_params, opt_state, batch, keys)
            avg_elbo_epoch += summed_elbo_values
        avg_elbos.append(avg_elbo_epoch / num_seqs)

    return q_params, jnp.array(avg_elbos)


fitted_q_params, avg_elbos = fit(q_params)


plt.plot(avg_elbos, label='$\mathcal{L}(\\theta,\\phi)$')
plt.axhline(y=avg_evidence, c='red', label = '$log p_{\\theta}(x)$' )
plt.xlabel('Epoch') 
plt.tight_layout()
plt.title(f'Epoch = {num_seqs} sequences of {seq_length} observations, $d_Z = {state_dim}, d_X = {obs_dim}$')
plt.legend()
plt.autoscale(True)
plt.show()
#%% Visualing result q against p 

smoothed_means_kalman, smoothed_covs_kalman = jax.vmap(kalman_smooth, in_axes=(0, None))(obs_seqs, p)
import numpy as np 

# get_marginals = lambda obs_seq: NonLinearELBO(p_model, QFromBackward(q_model)).compute_tractable_terms(obs_seq, p, fitted_q_params)[1]
get_marginals = lambda obs_seq: NonLinearELBO(p_model, QFromForward(q_model)).compute_tractable_terms(obs_seq, p, hmm.GaussianHMM.build_from_dict(fitted_q_params, q_model))[1]
smoothed_means, smoothed_covs = jax.vmap(get_marginals)(obs_seqs)
print('MSE smoothed with q_phi:',jnp.mean((smoothed_means - state_seqs)**2))
print('MSE smoothed with Kalman:',jnp.mean((smoothed_means_kalman - state_seqs)**2))


def plot_relative_errors(seq_nb):
    time_axis = range(seq_length)
    means_kalman, covs_kalman = smoothed_means_kalman[seq_nb].squeeze(), smoothed_covs_kalman[seq_nb].squeeze()
    means, covs = smoothed_means[seq_nb].squeeze(), smoothed_covs[seq_nb].squeeze()
    true_states = state_seqs[seq_nb]
    fig, (ax0, ax1) = plt.subplots(1,2, sharey=True, figsize=(12,5))
    ax0.errorbar(x=time_axis, fmt = '_', y=means_kalman, yerr=1.96 * np.sqrt(covs_kalman), label='Smoothed z, $1.96\\sigma$')
    ax0.scatter(x=time_axis, marker = '_', y=true_states, c='r', label='True z')
    ax0.set_title('Kalman')
    ax1.errorbar(x=time_axis, fmt = '_', y=means, yerr=1.96 * np.sqrt(covs), label='Smoothed z, $1.96\\sigma$')
    ax1.scatter(x=time_axis, marker = '_', y=true_states, c='r', label='True z')
    ax1.set_title('Backward variational')
    plt.suptitle(f'Sequence {seq_nb}')
    plt.legend()
    plt.tight_layout()
    plt.autoscale(True)
    plt.show()

for seq_nb in range(2):
    plot_relative_errors(seq_nb)

#%%
# import numpy as np
# from matplotlib.patches import Ellipse
# import matplotlib.transforms as transforms

# def visualize_predictions_on_sequence(states, predicted_means, predicted_covs):

#     for state, predicted_mean, predicted_cov in zip(states, predicted_means, predicted_covs): 
#         plt.scatter(state[0],state[1], c='r')
#         plt.scatt er(predicted_mean[0], predicted_mean[1], c='tab:purple')
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