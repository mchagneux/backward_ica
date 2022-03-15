#%% Imports 
from functools import partial
from jax import numpy as jnp, random, config, jit, vmap, value_and_grad
import optax

import matplotlib.pyplot as plt
from backward_ica.hmm import GaussianHMM

config.update("jax_enable_x64", True)
key = random.PRNGKey(0)


from backward_ica.kalman import filter as kalman_filter, smooth as kalman_smooth
from backward_ica.elbo import linear_gaussian_elbo

#%% Generate dataset
state_dim, obs_dim = 2, 3
num_sequences = 30
sequences_length = 16

key, subkey = random.split(key, 2)

p_raw, p_aux_info = GaussianHMM.get_random_params(subkey, state_dim, obs_dim, 
                                                transition_mapping_type='linear', 
                                                emission_mapping_type='linear')

hmm = GaussianHMM.build_from_dict(p_raw, p_aux_info)

key, *subkeys = random.split(key, num_sequences+1)
state_samples, obs_samples = vmap(hmm.sample, in_axes=(0, None))(jnp.array(subkeys), sequences_length)

evidence_sequences = vmap(lambda observations, hmm: kalman_filter(observations, hmm)[-1], in_axes=(0, None))(obs_samples, hmm)
elbo_sequences = vmap(linear_gaussian_elbo, in_axes=(0, None, None, None, None))(obs_samples, p_raw, p_raw, p_aux_info, p_aux_info)

mean_evidence = jnp.mean(evidence_sequences)
print('Sanity check ELBO computation:',jnp.abs(jnp.sum(evidence_sequences) - jnp.sum(elbo_sequences)))

#%% Optimization
key, subkey = random.split(key, 2)
q_raw, q_aux_info = GaussianHMM.get_random_params(subkey, state_dim, obs_dim, 
                                                transition_mapping_type='linear', emission_mapping_type='linear')

loss = partial(linear_gaussian_elbo, p_aux_info=p_aux_info, q_aux_info=q_aux_info)
optimizer = optax.adam(learning_rate=-1e-3)

@jit
def step(p_raw, q_raw, opt_state, batch):
    loss_value, grads = value_and_grad(loss, argnums=2)(batch, p_raw, q_raw)
    updates, opt_state = optimizer.update(grads, opt_state, q_raw)
    q_raw = optax.apply_updates(q_raw, updates)
    return p_raw, q_raw, opt_state, -loss_value



def fit(p_raw, q_raw, optimizer: optax.GradientTransformation) -> optax.Params:
    opt_state = optimizer.init(q_raw)

    eps = jnp.inf
    old_mean_epoch_elbo = - jnp.mean(vmap(loss, in_axes=(0,None,None))(obs_samples, p_raw, q_raw))
    epoch_nb = 0
    mean_elbos = [old_mean_epoch_elbo - mean_evidence]
    while eps > 1e-2:
    # for _ in range(10):
        epoch_elbo = 0.0
        for batch in obs_samples: 
            p_raw, q_raw, opt_state, elbo_value = step(p_raw, q_raw, opt_state, batch)
            epoch_elbo += elbo_value
        mean_epoch_elbo = epoch_elbo/len(obs_samples)
        
        eps = jnp.abs(mean_epoch_elbo - old_mean_epoch_elbo)
        epoch_nb+=1
        mean_elbos.append(mean_epoch_elbo - mean_evidence)
        old_mean_epoch_elbo = mean_epoch_elbo
    return q_raw, mean_elbos

fitted_q_raw, mean_elbos = fit(p_raw, q_raw, optimizer)

plt.plot(mean_elbos)
plt.xlabel('Epoch nb'), 
plt.ylabel('$|\mathcal{L}(\\theta,\\phi)- \log p_\\theta(x)|$')
plt.show()

q_hmm = GaussianHMM.build_from_dict(fitted_q_raw, q_aux_info)

def squared_error_expectation_against_true_states(states, observations, approximate_linear_gaussian_model, additive_functional):
    smoothed_states, _ = kalman_smooth(observations, approximate_linear_gaussian_model)
    return jnp.sqrt((additive_functional(smoothed_states) - additive_functional(states)) ** 2)

additive_functional = partial(jnp.sum, axis=0)
mse_in_expectations = vmap(squared_error_expectation_against_true_states, in_axes=(0,0, None, None))
print('Smoothed with q:', jnp.mean(mse_in_expectations(state_samples, obs_samples, q_hmm, additive_functional), axis=0))
print('Smoothed with p:', jnp.mean(mse_in_expectations(state_samples, obs_samples, hmm, additive_functional), axis=0))
#%% Trying the nonlinear case 

