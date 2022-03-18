#%% Imports 
from functools import partial
from jax import numpy as jnp, random, config, jit, vmap, value_and_grad
import optax

import matplotlib.pyplot as plt
import backward_ica.hmm as hmm 

key = random.PRNGKey(0)


from backward_ica.kalman import filter as kalman_filter, smooth as kalman_smooth
from backward_ica.elbo import get_elbo
config.update("jax_enable_x64", True)

#%% Generate dataset
state_dim, obs_dim = 2, 3
num_sequences = 16
sequences_length = 30

key, subkey = random.split(key, 2)

p_params, p_def = hmm.get_random_params(subkey, state_dim, obs_dim, 
                                        transition_mapping_type='linear', 
                                        emission_mapping_type='linear')

p = hmm.GaussianHMM.build_from_dict(p_params, p_def)

key, *subkeys = random.split(key, num_sequences+1)
state_samples, obs_samples = vmap(p.sample, in_axes=(0, None))(jnp.array(subkeys), sequences_length)

elbo = get_elbo(p_def, p_def)
elbo_sequences = vmap(elbo, in_axes=(0, None, None))(obs_samples, p_params, p_params)
evidence_sequences = vmap(lambda sequence, p: kalman_filter(sequence, p)[-1], in_axes=(0, None))(obs_samples, p)
mean_evidence = jnp.mean(evidence_sequences)

print('Sanity check ELBO computation:',jnp.mean(jnp.abs(elbo_sequences - evidence_sequences)))

#%% Optimization
key, subkey = random.split(key, 2)
q_params, q_def = hmm.get_random_params(subkey, state_dim, obs_dim, 
                                                transition_mapping_type='linear', emission_mapping_type='linear')

loss = get_elbo(p_def, q_def)
optimizer = optax.adam(learning_rate=-1e-3)

@jit
def step(p_params, q_params, opt_state, batch):
    loss_value, grads = value_and_grad(loss, argnums=2)(batch, p_params, q_params)
    updates, opt_state = optimizer.update(grads, opt_state, q_params)
    q_params = optax.apply_updates(q_params, updates)
    return p_params, q_params, opt_state, loss_value



def fit(p_params, q_params, optimizer: optax.GradientTransformation) -> optax.Params:
    opt_state = optimizer.init(q_params)

    eps = jnp.inf
    old_mean_elbo = jnp.mean(vmap(loss, in_axes=(0,None,None))(obs_samples, p_params, q_params))
    epoch_nb = 0
    eps = jnp.inf
    dist_to_evidence = [old_mean_elbo - mean_evidence]
    while eps > 1e-2:
    # for _ in range(10):
        epoch_elbo = 0.0
        for batch in obs_samples: 
            p_params, q_params, opt_state, elbo_value = step(p_params, q_params, opt_state, batch)
            epoch_elbo += elbo_value
        mean_elbo = epoch_elbo/obs_samples.shape[0]
        
        eps = jnp.abs(mean_elbo - old_mean_elbo)
        epoch_nb+=1
        dist_to_evidence.append(mean_elbo - mean_evidence)
        old_mean_elbo = mean_elbo
    return q_params, dist_to_evidence

fitted_q_params, dist_to_evidence = fit(p_params, q_params, optimizer)

plt.plot(dist_to_evidence)
plt.xlabel('Epoch nb'), 
plt.ylabel('$|\mathcal{L}(\\theta,\\phi)- \log p_\\theta(x)|$')
plt.show()

q = hmm.GaussianHMM.build_from_dict(fitted_q_params, q_def)

def squared_error_expectation_against_true_states(states, observations, approximate_linear_gaussian_model, additive_functional):
    smoothed_states, _ = kalman_smooth(observations, approximate_linear_gaussian_model)
    return jnp.sqrt((additive_functional(smoothed_states) - additive_functional(states)) ** 2)

additive_functional = partial(jnp.sum, axis=0)
mse_in_expectations = vmap(squared_error_expectation_against_true_states, in_axes=(0,0, None, None))
print('Smoothed with q:', jnp.mean(mse_in_expectations(state_samples, obs_samples, q, additive_functional), axis=0))
print('Smoothed with p:', jnp.mean(mse_in_expectations(state_samples, obs_samples, p, additive_functional), axis=0))