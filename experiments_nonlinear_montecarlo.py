#%%
from collections import namedtuple
from backward_ica.elbo import NonLinearELBO, QFromBackward, QFromForward
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
from backward_ica.utils import LinearGaussianKernel, _mappings
config.update("jax_enable_x64", True)
import copy
#%% Hyperparameters 
seed_model_params = 43
seed_infer = 56
num_starting_points = 10
state_dim, obs_dim = 1,2
seq_length = 32
num_seqs = 4092

batch_size = 8
learning_rate = 1e-3
num_epochs = 50
num_batches_per_epoch = num_seqs // batch_size

q_forward_linear_gaussian = False 
use_true_backward_update = False
key = jax.random.PRNGKey(seed_model_params)
infer_key = jax.random.PRNGKey(seed_infer)
#%% Define p 

key, *subkeys = jax.random.split(key, 3)
p_params, p_model = hmm.get_random_params(subkeys[0], subkeys[1], 
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
# elbo_on_seqs_with_true_model = lambda obs_seq : -NonLinearELBO(p_model, QFromForward(p_model)).compute(obs_seq, None, p_params, p_params)
# print('Average ELBO with q=p',jnp.mean(jax.vmap(elbo_on_seqs_with_true_model)(obs_seqs)))
#%% Define q 

d = state_dim

def backwd_update(filt_mean, filt_cov):
    net = hk.nets.MLP((8, d**2 + d + d*(d+1) // 2))
    out = net(jnp.concatenate((filt_mean, jnp.tril(filt_cov).flatten())))
    A = out[:d**2].reshape((d,d))
    a = out[d**2:d**2+d]
    cov = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(out[d**2+d:])
    return A, a, cov @ cov.T

def filt_update(obs, filt_mean, filt_cov):

    net = hk.nets.MLP((8, d + d*(d+1) // 2))
    out = net(jnp.concatenate((obs, filt_mean, jnp.tril(filt_cov).flatten())))
    mean = out[:d]
    cov_chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(out[d:])
    return mean, cov_chol @ cov_chol.T

filt_update_init, filt_update_apply = hk.without_apply_rng(hk.transform(filt_update))
backward_init, backward_apply = hk.without_apply_rng(hk.transform(backwd_update))


dummy_obs = obs_seqs[0][0]
dummy_mean = jnp.empty((state_dim,))
dummy_cov = jnp.empty((state_dim, state_dim))

if q_forward_linear_gaussian:
    Q = QFromForward
    q_model = p_model
else: 
    if use_true_backward_update:
        dummy_namedtuple = namedtuple('dummy_namedtuple',['transition'])
        def backward_apply(shared_param, filt_mean, filt_cov, params):
            weight = jnp.diag(params['mapping_params']['weight'])
            bias = params['mapping_params']['bias']
            cov = jnp.diag(jnp.exp(params['cov_params']['cov']))
            params = LinearGaussianKernel(_mappings['linear'],
                                        {'weight':weight,'bias':bias},
                                        # cov,
                                        *prec_and_det(cov))
            return hmm.update_backward(Gaussian(filt_mean, filt_cov, *prec_and_det(filt_cov)), dummy_namedtuple(transition=params))[:-1]
    Q = QFromBackward
    q_model = {'filtering':{'update':filt_update_apply},
        'backward':backward_apply}

other_key = jax.random.PRNGKey(1)

def init_q(infer_subkey):

    if q_forward_linear_gaussian: 
        q_params, _ = hmm.get_random_params(infer_subkey, other_key, state_dim, obs_dim, 'linear','linear')
        
    else:
        subkeys = jax.random.split(infer_subkey, 3)
        # shared_param = jax.random.uniform(subkeys[0],(1000,))
        filt_update_params = filt_update_init(subkeys[1], dummy_obs, dummy_mean, dummy_cov)

        if use_true_backward_update:
            backward_params = hmm.get_random_params(subkeys[2], 
                                        state_dim, 
                                        obs_dim,
                                        transition_mapping_type='linear',
                                        emission_mapping_type='linear')[0]['transition']
        else:
            backward_params = backward_init(subkeys[2], dummy_mean, dummy_cov)



        q_params = {'filtering':{'update':filt_update_params},
                    'backward':backward_params}


    return q_params

#%% Fit q

elbo = NonLinearELBO(p_model, Q(q_model)).compute
loss = lambda obs_seq, key, q_params: elbo(obs_seq, key, p_params, q_params)
optimizer = optax.adam(learning_rate=learning_rate)


def fit(init_q_params, subkey_montecarlo=None):
    opt_state = optimizer.init(init_q_params)

    subkeys = jnp.empty((num_epochs, num_seqs, 1))
    # subkeys = jax.random.split(subkey_montecarlo, num_seqs * num_epochs)
    # subkeys = jnp.array(subkeys).reshape(num_epochs,num_seqs,-1)

    @jax.jit
    def batch_step(carry, xs):

        def q_step(q_params, opt_state, batch, keys):
            neg_elbo_values, grads = jax.vmap(jax.value_and_grad(loss, argnums=2), in_axes=(0,0,None))(batch, keys, q_params)
            avg_grads = jax.tree_util.tree_map(jnp.mean, grads)
            updates, opt_state = optimizer.update(avg_grads, opt_state, q_params)
            q_params = optax.apply_updates(q_params, updates)
            return q_params, opt_state, jnp.mean(-neg_elbo_values)

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

    (fitted_q_params, _), avg_elbos = jax.lax.scan(epoch_step, init=(init_q_params, opt_state), xs=subkeys)
    print('Last avg elbo value',avg_elbos[-1])

    return fitted_q_params, avg_elbos

all_fitted_q_params = []
all_avg_elbos = []

def plot_relative_errors(smoothed_means_kalman, smoothed_covs_kalman, smoothed_means, smoothed_covs, seq_nb):
    time_axis = range(seq_length)
    means_kalman, covs_kalman = smoothed_means_kalman[seq_nb].squeeze(), smoothed_covs_kalman[seq_nb].squeeze()
    means, covs = smoothed_means[seq_nb].squeeze(), smoothed_covs[seq_nb].squeeze()
    print('Marginal smoothing cov at time 0:',covs[0])
    true_states = state_seqs[seq_nb]
    ax0 = plt.gcf().add_subplot(232)
    ax0.errorbar(x=time_axis, fmt = '_', y=means_kalman, yerr=1.96 * np.sqrt(covs_kalman), label='Smoothed z, $1.96\\sigma$')
    ax0.scatter(x=time_axis, marker = '_', y=true_states, c='r', label='True z')
    ax0.set_title('Kalman')
    ax0.set_xlabel('t')
    ax1 = plt.gcf().add_subplot(233, sharey=ax0)
    ax1.errorbar(x=time_axis, fmt = '_', y=means, yerr=1.96 * np.sqrt(covs), label='Smoothed z, $1.96\\sigma$')
    ax1.scatter(x=time_axis, marker = '_', y=true_states, c='r', label='True z')
    ax1.set_title('Backward variational')
    ax1.set_xlabel('t')


def visualize_inferred_states(fitted_q_params):

    get_marginals = lambda obs_seq: Q(q_model).marginals(obs_seq, fitted_q_params, p)
    smoothed_means, smoothed_covs = jax.vmap(get_marginals)(obs_seqs)
    smoothed_means_kalman, smoothed_covs_kalman = jax.vmap(kalman_smooth, in_axes=(0, None))(obs_seqs, p)
    
    # smoothed_means, smoothed_covs = jax.vmap(kalman_smooth, in_axes=(0, None))(obs_seqs, hmm.GaussianHMM.build_from_dict(fitted_q_params, q_model))
    print('MSE smoothed with q_phi:',jnp.mean((smoothed_means - state_seqs)**2))
    print('MSE smoothed with Kalman:',jnp.mean((smoothed_means_kalman - state_seqs)**2))


    plot_relative_errors(smoothed_means_kalman, smoothed_covs_kalman, smoothed_means, smoothed_covs, 3)

for infer_subkey in jax.random.split(infer_key, num_starting_points):
    init_q_params = init_q(infer_subkey)
    fitted_q_params, avg_elbos = fit(init_q_params)
    all_fitted_q_params.append(copy.deepcopy(fitted_q_params))
    all_avg_elbos.append(copy.deepcopy(all_avg_elbos))
    fig = plt.figure(figsize=(20,10))
    ax0 = fig.add_subplot(231)
    ax0.plot(avg_elbos, label='$\mathcal{L}(\\theta,\\phi)$')
    ax0.axhline(y=avg_evidence, c='red', label = '$log p_{\\theta}(x)$' )
    ax0.set_xlabel('Epoch') 
    ax0.set_title(f'{num_seqs} seqs of {seq_length} obs$')
    visualize_inferred_states(fitted_q_params)
    plt.autoscale(True)
    plt.legend()
    plt.show()
#%%