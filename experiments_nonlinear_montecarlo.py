#%%
from collections import namedtuple
from backward_ica.elbo import LinearELBO, NonLinearELBO, QFromBackward, QFromForward
import backward_ica.hmm as hmm
from backward_ica.kalman import filter as kalman_filter, smooth as kalman_smooth, predict as kalman_predict, update as kalman_update
import jax 
import jax.numpy as jnp
import haiku as hk 
import optax 
import numpy as np 
from jax import config 
import matplotlib.pyplot as plt
from  backward_ica.utils import Gaussian, prec_and_det

from backward_ica.utils import LinearGaussianKernel
config.update("jax_enable_x64", True)

#%% Hyperparameters 
seed = 54

state_dim, obs_dim = 1,2
seq_length = 32
num_seqs = 512

batch_size = 8
learning_rate = 1e-3
num_epochs = 500

q_forward_linear_gaussian = False 
use_true_predict = False 
use_true_update = False 
use_true_backward_update = False 
key = jax.random.PRNGKey(seed)
#%% Define p 

key, subkey = jax.random.split(key,2)
p_params, p_model = hmm.get_random_params(subkey, 
                                    state_dim, 
                                    obs_dim,
                                    transition_mapping_type='linear',
                                    emission_mapping_type='linear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_model)
p_copy = hmm.GaussianHMM.build_from_dict(p_params, p_model)

key, *subkeys = jax.random.split(key, num_seqs+1)
state_seqs, obs_seqs = jax.vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), seq_length)

kalman_log_likel = lambda obs_seq: kalman_filter(obs_seq, p)[-1]
avg_evidence = jnp.mean(jax.vmap(kalman_log_likel)(obs_seqs))
print('Average log-evidence:',avg_evidence)
#%% Define q 

d = state_dim

if not q_forward_linear_gaussian: 

    def backwd(filt_mean, filt_cov):
        net = hk.nets.MLP((8, d**2 + d + d*(d+1) // 2))
        out = net(jnp.concatenate((filt_mean, jnp.tril(filt_cov).flatten())))
        A = out[:d**2].reshape((d,d))
        a = out[d**2:d**2+d]
        cov = jnp.zeros((d,d))
        cov = cov.at[jnp.tril_indices(d)].set(out[d**2+d:])
        return A, a, cov @ cov.T

    def filt_predict(filt_mean, filt_cov):
        net = hk.nets.MLP((8, d + d*(d+1) // 2))
        out = net(jnp.concatenate((filt_mean, jnp.tril(filt_cov).flatten())))
        mean = out[:d]
        cov = jnp.zeros((d,d))
        cov = cov.at[jnp.tril_indices(d)].set(out[d:])
        return mean, cov @ cov.T

    def filt_update(obs, pred_mean, pred_cov):
        net = hk.nets.MLP((8, d + d*(d+1) // 2))
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
    
    if use_true_predict: 
        filt_predict_apply = lambda filt_mean, filt_cov, params:kalman_predict(filt_mean, filt_cov, params)
        filt_predict_params = p_copy.transition
    
    if use_true_update:
        filt_update_apply = lambda obs, pred_mean, pred_cov, params: kalman_update(pred_mean, pred_cov, obs, params)
        filt_update_params = p_copy.emission 
    
    if use_true_backward_update:
        backward_apply = lambda filt_mean, filt_cov, params: hmm.update_backward(Gaussian(filt_mean, filt_cov, *prec_and_det(filt_cov)), params)[:-1]
        dummy_namedtuple = namedtuple('dummy_named_tuple',['transition'])
        backward_params = dummy_namedtuple(p_copy.transition)
    
    q_model = {'filtering':{'predict':filt_predict_apply, 'update':filt_update_apply},
            'backward':backward_apply}

    q_params = {'filtering':{'predict':filt_predict_params, 'update':filt_update_params},
                'backward':backward_params}

    Q = QFromBackward

else: 
    key, subkey = jax.random.split(key, 2)
    q_params, q_model = hmm.get_random_params(subkey, state_dim, obs_dim, 'linear','linear')
    Q = QFromForward


#%% Fit q

elbo = NonLinearELBO(p_model, Q(q_model)).compute
loss = lambda obs_seq, key, q_params: elbo(obs_seq, key, p_params, q_params)

optimizer = optax.adam(learning_rate=learning_rate)

batch_size = 8
num_batches_per_epoch = num_seqs // batch_size

def q_step(q_params, opt_state, batch, keys):
    neg_elbo_values, grads = jax.vmap(jax.value_and_grad(loss, argnums=2), in_axes=(0,0,None))(batch, keys, q_params)
    avg_grads = jax.tree_util.tree_map(jnp.mean, grads)
    updates, opt_state = optimizer.update(avg_grads, opt_state, q_params)
    q_params = optax.apply_updates(q_params, updates)
    return q_params, opt_state, jnp.mean(-neg_elbo_values)

# key, *subkeys = jax.random.split(key, num_seqs * num_epochs + 1)
# subkeys = jnp.array(subkeys).reshape(num_epochs,num_seqs,-1)
subkeys = jnp.empty((num_epochs, num_seqs, 1))

def fit(q_params):
    opt_state = optimizer.init(q_params)

    @jax.jit
    def batch_step(carry, xs):
        q_params, opt_state, subkeys_epoch = carry
        batch_start = xs
        batch_obs_seq = jax.lax.dynamic_slice_in_dim(obs_seqs, batch_start, batch_size)
        batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, batch_size)
        q_params, opt_state, avg_elbo_batch = q_step(q_params, opt_state, batch_obs_seq, batch_keys)
        return (q_params, opt_state, subkeys_epoch), avg_elbo_batch

    
    def epoch_step(carry, xs):
        q_params, opt_state = carry
        subkeys_epoch = xs
        batch_start_indices = jnp.arange(0, num_seqs, batch_size)
        (q_params, opt_state, _), avg_elbo_batches = jax.lax.scan(batch_step, 
                                                            init=(q_params, opt_state, subkeys_epoch), 
                                                            xs = batch_start_indices)
        return (q_params, opt_state), jnp.mean(avg_elbo_batches)

    (q_params, _), avg_elbos = jax.lax.scan(epoch_step, init=(q_params, opt_state), xs=subkeys)

    return q_params, avg_elbos


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


get_marginals = lambda obs_seq: Q(q_model).marginals(obs_seq, fitted_q_params, p.prior)
smoothed_means, smoothed_covs = jax.vmap(get_marginals)(obs_seqs)

smoothed_means_kalman, smoothed_covs_kalman = jax.vmap(kalman_smooth, in_axes=(0, None))(obs_seqs, p)

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