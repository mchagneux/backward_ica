import backward_ica.hmm as hmm 
from jax import random, numpy as jnp, vmap, tree_util, value_and_grad, jit
import matplotlib.pyplot as plt 
from backward_ica.kalman import filter as kalman_filter, smooth as kalman_smooth
from functools import partial

import optax 
from backward_ica.elbo import NonLinearELBO
import haiku as hk

key = random.PRNGKey(0)
key, *subkeys = random.split(key, 3)
state_dim, obs_dim = 2, 3
num_sequences = 50
sequences_length = 4
#%% Trying the nonlinear case 

p_params, p_def = hmm.get_random_params(subkeys[0], state_dim, obs_dim, 
                                        transition_mapping_type='linear', 
                                        emission_mapping_type='nonlinear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_def)


key, *subkeys = random.split(key, num_sequences+1)
state_samples, obs_samples = vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), sequences_length)

# evidence_function = lambda sequence, p: kalman_filter(sequence, p)[-1]
# evidence_sequences = vmap(evidence_function, in_axes=(0, None))(obs_samples, p)
# mean_evidence = jnp.mean(evidence_sequences)


dim_z = p.transition.cov.shape[0]
def rec_net_spec(x):
    net = hk.nets.MLP((64,2*dim_z))
    out = net(x)
    v = out[:dim_z]
    log_W_diag = out[dim_z:]
    return v, -jnp.diag(jnp.exp(log_W_diag))

def backward_spec(filtering_mean, filtering_cov):

    net = hk.nets.MLP((64, 3*dim_z))
    out = net(jnp.concatenate((filtering_mean, jnp.diagonal(filtering_cov))))
    A = jnp.diag(out[:dim_z])
    a = out[dim_z:2*dim_z]
    log_cov_diag = out[2*dim_z:]
    return A, a, jnp.diag(jnp.exp(log_cov_diag))

def filtering_init_spec(observation):
    net = hk.nets.MLP((64, 2*dim_z))
    out = net(observation)
    mean = out[:dim_z]
    log_cov_diag = out[dim_z:]
    return mean, jnp.diag(jnp.exp(log_cov_diag))

def filtering_update_spec(observation, filtering_mean, filtering_cov):
    net = hk.nets.MLP((64, 2*dim_z))
    out = net(jnp.concatenate((observation, filtering_mean, jnp.diagonal(filtering_cov))))
    mean = out[:dim_z]
    log_cov_diag = out[dim_z:]
    return mean, jnp.diag(jnp.exp(log_cov_diag))

key, *subkey = random.split(key, 5)

rec_net_init, rec_net_def = hk.without_apply_rng(hk.transform(rec_net_spec))
filtering_init_init, filtering_init_def = hk.without_apply_rng(hk.transform(filtering_init_spec))
filtering_update_init, filtering_update_def = hk.without_apply_rng(hk.transform(filtering_update_spec))
backward_init, backward_def = hk.without_apply_rng(hk.transform(backward_spec))

dummy_obs = obs_samples[0][0]
dummy_mean = jnp.empty((dim_z,))
dummy_cov = jnp.empty((dim_z, dim_z))

rec_net_params = rec_net_init(subkeys[0], dummy_obs)
filtering_init_params = filtering_init_init(subkeys[1], dummy_obs)
filtering_update_params = filtering_update_init(subkeys[2], dummy_obs, dummy_mean, dummy_cov)
backward_params = backward_init(subkeys[3], dummy_mean, dummy_cov)


q_def = {'filtering':{'init':filtering_init_def,'update':filtering_update_def},
        'backward':backward_def}

q_params = {'filtering':{'init':filtering_init_params,'update':filtering_update_params},
        'backward':backward_params}

elbo = NonLinearELBO(p_def, q_def, rec_net_def).compute

loss = lambda observations, q_params, rec_net_params: elbo(observations, p_params, q_params, rec_net_params)
# print(loss(obs_samples[0], aux_params))

# print(loss(obs_samples[0], q_params, rec_net_params))

optimizer = optax.adam(learning_rate=1e-3)
optimizer_rec_net = optax.adam(learning_rate=1e-3)

@jit
def q_step(q_params, rec_net_params, opt_state, batch):
    loss_values, grads = vmap(value_and_grad(loss, argnums=1), in_axes=(0,None,None))(batch, q_params, rec_net_params)
    avg_loss_value = jnp.mean(loss_values)
    avg_grads = tree_util.tree_map(jnp.mean, grads)
    updates, opt_state = optimizer.update(avg_grads, opt_state, q_params)
    q_params = optax.apply_updates(q_params, updates)
    return q_params, opt_state, avg_loss_value

@jit
def rec_net_step(q_params, rec_net_params, opt_state, batch):
    loss_values, grads = vmap(value_and_grad(loss, argnums=2), in_axes=(0,None,None))(batch, q_params, rec_net_params)
    avg_loss_value = jnp.mean(loss_values)
    avg_grads = tree_util.tree_map(jnp.mean, grads)
    updates, opt_state = optimizer_rec_net.update(avg_grads, opt_state, rec_net_params)
    rec_net_params = optax.apply_updates(rec_net_params, updates)
    return rec_net_params, opt_state, avg_loss_value


def fit(q_params, rec_net_params) -> optax.Params:
    q_opt_state = optimizer.init(q_params)
    rec_net_opt_state = optimizer_rec_net.init(rec_net_params)
    avg_neg_elbos = []
    for _ in range(30):
        for _ in range(300):
            q_params, q_opt_state, avg_neg_elbo = q_step(q_params, rec_net_params, q_opt_state, obs_samples)
            avg_neg_elbos.append(avg_neg_elbo)

        rec_net_params, rec_net_opt_state, avg_neg_elbo = rec_net_step(q_params, rec_net_params, rec_net_opt_state, obs_samples)
        avg_neg_elbos.append(avg_neg_elbo)
    return q_params, rec_net_params, avg_neg_elbos

# params = rec_net_params
fitted_q_params, fitted_aux_params, avg_neg_elbos = fit(q_params, rec_net_params)

plt.plot(avg_neg_elbos)
plt.xlabel('Epoch nb'), 
plt.ylabel('$-\mathcal{L}(\\theta,\\phi)$')
plt.show()

q = hmm.GaussianHMM.build_from_dict(fitted_q_params, q_def)

def squared_error_expectation_against_true_states(states, observations, approximate_linear_gaussian_model, additive_functional):
    smoothed_states, _ = kalman_smooth(observations, approximate_linear_gaussian_model)
    return (additive_functional(smoothed_states) - additive_functional(states)) ** 2

additive_functional = partial(jnp.sum, axis=0)
mse_in_expectations = vmap(squared_error_expectation_against_true_states, in_axes=(0,0, None, None))
print('Smoothed with q:', jnp.mean(mse_in_expectations(state_samples, obs_samples, q, additive_functional)))