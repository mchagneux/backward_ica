#%% Imports 
from audioop import avg
from functools import partial
from jax import numpy as jnp, random, config, jit, vmap, value_and_grad, tree_util
import optax

import matplotlib.pyplot as plt
import backward_ica.hmm as hmm 

key = random.PRNGKey(0)


from backward_ica.kalman import filter as kalman_filter, smooth as kalman_smooth
from backward_ica.elbo import get_neg_elbo
config.update("jax_enable_x64", True)

#%% Generate dataset
state_dim, obs_dim = 2, 3
num_sequences = 32
sequences_length = 16
batch_size = 1

key, subkey = random.split(key, 2)

p_params, p_def = hmm.get_random_params(subkey, state_dim, obs_dim, 
                                        transition_mapping_type='linear', 
                                        emission_mapping_type='linear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_def)

key, *subkeys = random.split(key, num_sequences+1)
state_samples, obs_samples = vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), sequences_length)

elbo = get_neg_elbo(p_def, p_def)
elbo_sequences = vmap(elbo, in_axes=(0, None, None))(obs_samples, p_params, p_params)
evidence_sequences = vmap(lambda sequence, p: kalman_filter(sequence, p)[-1], in_axes=(0, None))(obs_samples, p)
mean_evidence = jnp.mean(evidence_sequences)

print('Sanity check ELBO computation:',jnp.mean(jnp.abs(-elbo_sequences - evidence_sequences)))

#%% Optimization
key, subkey = random.split(key, 2)
q_params, q_def = hmm.get_random_params(subkey, state_dim, obs_dim, 
                                        transition_mapping_type='linear',
                                        emission_mapping_type='linear')

loss = lambda obs, q_params: get_neg_elbo(p_def, q_def)(obs, p_params, q_params)

optimizer = optax.adam(learning_rate=1e-1)

@jit
def q_step(q_params, opt_state, batch):
    loss_values, grads = vmap(value_and_grad(loss, argnums=1), in_axes=(0,None))(batch, q_params)
    avg_grads = tree_util.tree_map(jnp.mean, grads)
    updates, opt_state = optimizer.update(avg_grads, opt_state, q_params)
    q_params = optax.apply_updates(q_params, updates)
    return q_params, opt_state, jnp.mean(loss_values)

def loader(data):
    for index in range(0, len(data), batch_size):
        yield data[index:index+batch_size]


def fit(q_params, optimizer: optax.GradientTransformation) -> optax.Params:

    opt_state = optimizer.init(q_params)
    epoch_avg_neg_elbos = [0,1]
    num_batches = num_sequences // batch_size
    for _ in range(5000):
        avg_neg_elbo_epoch = 0.0
        for batch in loader(obs_samples):
            q_params, opt_state, avg_neg_elbo_batch = q_step(q_params, opt_state, batch)
            avg_neg_elbo_epoch += avg_neg_elbo_batch // num_batches
        epoch_avg_neg_elbos.append(avg_neg_elbo_epoch)
    return q_params, epoch_avg_neg_elbos[2:]

fitted_q_params, avg_neg_elbos = fit(q_params, optimizer)

plt.plot(avg_neg_elbos)
plt.xlabel('Epoch nb'), 
plt.ylabel('$|\mathcal{L}(\\theta,\\phi)- \log p_\\theta(x)|$')
plt.show()

q = hmm.GaussianHMM.build_from_dict(fitted_q_params, q_def)

def squared_error_expectation_against_true_states(states, observations, approximate_linear_gaussian_model, additive_functional):
    smoothed_states, _ = kalman_smooth(observations, approximate_linear_gaussian_model)
    return (additive_functional(smoothed_states) - additive_functional(states))**2

additive_functional = partial(jnp.sum, axis=0)
mse_in_expectations = vmap(squared_error_expectation_against_true_states, in_axes=(0,0, None, None))
print('Smoothed with q:', jnp.mean(mse_in_expectations(state_samples, obs_samples, q, additive_functional)))
print('Smoothed with p:', jnp.mean(mse_in_expectations(state_samples, obs_samples, p, additive_functional)))