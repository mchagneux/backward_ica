#%%
from functools import partial
from src.elbo import linear_gaussian_elbo
from src.hmm import LinearGaussianHMM
from src import kalman
from jax.random import PRNGKey
import jax.numpy as jnp
import jax
import optax
from jax import random
from src.misc import *
jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", True)
key = PRNGKey(0)
import matplotlib.pyplot as plt
#%% Checking that L(p,p) = log p(x)

state_dim, obs_dim = 2, 3
num_sequences = 20
length = 8

key, *subkeys = random.split(key,3)

p_raw = LinearGaussianHMM.get_random_model(key=subkeys[0], state_dim=state_dim, obs_dim=obs_dim)
q_raw = LinearGaussianHMM.get_random_model(key=subkeys[1], state_dim=state_dim, obs_dim=obs_dim)

p = actual_model_from_raw_parameters(p_raw)

linear_gaussian_sampler = jax.vmap(LinearGaussianHMM.sample_joint_sequence, in_axes=(0, None, None))
key, *subkeys = random.split(key, num_sequences+1)
state_sequences, obs_sequences = linear_gaussian_sampler(jnp.array(subkeys), p, length)


filter_obs_sequences = jax.vmap(kalman.filter, in_axes=(0, None))
elbo_sequences = jax.vmap(linear_gaussian_elbo, in_axes=(None, None, 0))

# average_evidence_across_sequences = jnp.mean(filter_obs_sequences(obs_sequences, p)[-1])
# average_elbo_across_sequences_with_true_model = jnp.mean(elbo_sequences(p_raw, p_raw, obs_sequences))
# print('Difference mean evidence Kalman and mean ELBO when q=p:', jnp.abs(average_evidence_across_sequences-average_elbo_across_sequences_with_true_model))


#%% Setting up optimizer 
def step(p_raw, q_raw, opt_state, batch):
    loss_value, grads = jax.value_and_grad(linear_gaussian_elbo, argnums=1)(p_raw, q_raw, batch)
    updates, opt_state = optimizer.update(grads, opt_state, q_raw)
    q_raw = optax.apply_updates(q_raw, updates)
    return p_raw, q_raw, opt_state, -loss_value
step = jax.jit(step)
q_raw = LinearGaussianHMM.get_random_model(key=subkeys[1], state_dim=state_dim, obs_dim=obs_dim)
average_elbo_across_sequences_with_init_q = jnp.mean(elbo_sequences(p_raw, q_raw, obs_sequences))
print('Different mean evidence and mean ELBO when q=q0:', jnp.abs(average_evidence_across_sequences-average_elbo_across_sequences_with_init_q))

#%% Running optim
optimizer = optax.adam(learning_rate=-1e-3)

def fit(p_raw, q_raw, optimizer: optax.GradientTransformation) -> optax.Params:
    opt_state = optimizer.init(q_raw)

    eps = jnp.inf
    old_mean_epoch_elbo = -average_elbo_across_sequences_with_init_q
    epoch_nb = 0
    mean_elbos = [old_mean_epoch_elbo - average_evidence_across_sequences]
    while eps > 1e-2:
        epoch_elbo = 0.0
        for batch in obs_sequences: 
            p_raw, q_raw, opt_state, elbo_value = step(p_raw, q_raw, opt_state, batch)
            epoch_elbo += elbo_value
        mean_epoch_elbo = epoch_elbo/len(obs_sequences)
        eps = jnp.abs(mean_epoch_elbo - old_mean_epoch_elbo)
        epoch_nb+=1
        mean_elbos.append(mean_epoch_elbo - average_evidence_across_sequences)
        old_mean_epoch_elbo = mean_epoch_elbo
    return q_raw, mean_elbos


q_raw, mean_elbos = fit(p_raw, q_raw, optimizer)

plt.plot(mean_elbos)
plt.xlabel('Epoch nb'), 
plt.ylabel('Mean of $\mathcal{L}(\\theta,\phi) - log p_{\\theta}$')

#%% Computing expectations 

q = actual_model_from_raw_parameters(q_raw)

def squared_error_expectation_against_true_states(states, observations, approximate_linear_gaussian_model, additive_functional):
    smoothed_states, _ = kalman.smooth(observations, approximate_linear_gaussian_model)
    return (additive_functional(smoothed_states) - additive_functional(states)) ** 2

additive_functional = partial(jnp.sum, axis=0)
mse_in_expectations = jax.vmap(squared_error_expectation_against_true_states, in_axes=(0,0, None, None))
print('MSE(E_q(h(z)), z_true):', jnp.mean(mse_in_expectations(state_sequences, obs_sequences, q, additive_functional), axis=0))
print('MSE(E_p(h(z)), z_true):', jnp.mean(mse_in_expectations(state_sequences, obs_sequences, p, additive_functional), axis=0))