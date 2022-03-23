import backward_ica.hmm as hmm 
from jax import random, numpy as jnp, vmap, tree_util, value_and_grad, jit
import matplotlib.pyplot as plt 
from backward_ica.kalman import filter as kalman_filter, smooth as kalman_smooth
from functools import partial

import optax 
from backward_ica.elbo import NonLinearELBO, get_neg_elbo
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

q_params, q_def = hmm.get_random_params(subkeys[1], state_dim, obs_dim, 
                                        transition_mapping_type='linear', 
                                        emission_mapping_type='linear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_def)

# q_params, q_def  = hmm.get_random_params(subkeys[1], state_dim, obs_dim, transition_mapping_type='linear', emission_mapping_type='linear')

key, *subkeys = random.split(key, num_sequences+1)
state_samples, obs_samples = vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), sequences_length)

# evidence_function = lambda sequence, p: kalman_filter(sequence, p)[-1]
# evidence_sequences = vmap(evidence_function, in_axes=(0, None))(obs_samples, p)
# mean_evidence = jnp.mean(evidence_sequences)


dim_z = p.transition.cov.shape[0]
def rec_net_forward(x):
    net = hk.nets.MLP((64,2*dim_z))
    out = net(x)
    v = out[:dim_z]
    log_W_diag = out[dim_z:]
    return v, -jnp.diag(jnp.exp(log_W_diag))

key, subkey = random.split(key, 2)
rec_net_init, rec_net_apply = hk.without_apply_rng(hk.transform(rec_net_forward))

aux_params = {'rec_net':rec_net_init(subkey, obs_samples[0][0])}
aux_defs = {'rec_net':rec_net_apply}

elbo = get_neg_elbo(p_def, q_def, aux_defs)

loss = lambda observations, q_params, aux_params: elbo(observations, p_params, q_params, aux_params)
# print(loss(obs_samples[0], aux_params))

optimizer = optax.adam(learning_rate=1e-3)
optimizer_aux = optax.adam(learning_rate=1e-4)

@jit
def q_step(q_params, aux_params, opt_state, batch):
    loss_values, grads = vmap(value_and_grad(loss, argnums=1), in_axes=(0,None,None))(batch, q_params, aux_params)
    avg_loss_value = jnp.mean(loss_values)
    avg_grads = tree_util.tree_map(jnp.mean, grads)
    updates, opt_state = optimizer.update(avg_grads, opt_state, q_params)
    q_params = optax.apply_updates(q_params, updates)
    return q_params, opt_state, avg_loss_value

@jit
def aux_step(q_params, aux_params, opt_state, batch):
    loss_values, grads = vmap(value_and_grad(loss, argnums=2), in_axes=(0,None,None))(batch, q_params, aux_params)
    avg_loss_value = jnp.mean(loss_values)
    avg_grads = tree_util.tree_map(jnp.mean, grads)
    updates, opt_state = optimizer_aux.update(avg_grads, opt_state, aux_params)
    aux_params = optax.apply_updates(aux_params, updates)
    return aux_params, opt_state, avg_loss_value


def fit(q_params, aux_params) -> optax.Params:
    q_opt_state = optimizer.init(q_params)
    aux_opt_state = optimizer_aux.init(aux_params)
    avg_neg_elbos = []
    for _ in range(50):
        for _ in range(100):
            q_params, q_opt_state, avg_neg_elbo = q_step(q_params, aux_params, q_opt_state, obs_samples)
            avg_neg_elbos.append(avg_neg_elbo)

        aux_params, aux_opt_state, avg_neg_elbo = aux_step(q_params, aux_params, aux_opt_state, obs_samples)
        avg_neg_elbos.append(avg_neg_elbo)
    return q_params, aux_params, avg_neg_elbos

# params = aux_params
fitted_q_params, fitted_aux_params, avg_neg_elbos = fit(q_params, aux_params)

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