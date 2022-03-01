#%% Imports 
from functools import partial
from jax import numpy as jnp, random, config, jit, vmap, value_and_grad
import optax

import matplotlib.pyplot as plt

config.update("jax_enable_x64", True)
key = random.PRNGKey(0)


from backward_ica.kalman import filter as kalman_filter, smooth as kalman_smooth
from backward_ica.elbo import linear_gaussian_elbo
from backward_ica.misc import parameters_from_raw_parameters
from backward_ica.hmm import LinearGaussianHMM

#%% Generate dataset
state_dim, obs_dim = 2, 3
num_sequences = 30
length = 16

key, *subkeys = random.split(key, 3)

p_raw = LinearGaussianHMM.get_random_model(key=subkeys[0], state_dim=state_dim, obs_dim=obs_dim)
q_raw = LinearGaussianHMM.get_random_model(key=subkeys[1], state_dim=state_dim, obs_dim=obs_dim)

p = parameters_from_raw_parameters(p_raw)

linear_gaussian_sampler = vmap(LinearGaussianHMM.sample_joint_sequence, in_axes=(0, None, None))
likelihood_via_kalman = lambda observations, model: kalman_filter(observations, model)[-1]

evidence_sequences = vmap(likelihood_via_kalman, in_axes=(0, None))
elbo_sequences = jit(vmap(linear_gaussian_elbo, in_axes=(None, None, 0)))

key, *subkeys = random.split(key, 1+num_sequences)
state_sequences, obs_sequences = linear_gaussian_sampler(jnp.array(subkeys), p, length)
average_evidence_dataset = jnp.mean(evidence_sequences(obs_sequences, p))

#%% Sanity check 
print('Difference elbo evidence when q=p:', jnp.mean(evidence_sequences(obs_sequences, p)) - \
            jnp.mean(elbo_sequences(p_raw, p_raw, obs_sequences)))

#%% Optimization 
optimizer = optax.adam(learning_rate=-1e-3)

@jit
def step(p_raw, q_raw, opt_state, batch):
    loss_value, grads = value_and_grad(linear_gaussian_elbo, argnums=1)(p_raw, q_raw, batch)
    updates, opt_state = optimizer.update(grads, opt_state, q_raw)
    q_raw = optax.apply_updates(q_raw, updates)
    return p_raw, q_raw, opt_state, -loss_value



def fit(p_raw, q_raw, optimizer: optax.GradientTransformation) -> optax.Params:
    opt_state = optimizer.init(q_raw)

    eps = jnp.inf
    old_mean_epoch_elbo = -jnp.mean(elbo_sequences(p_raw, q_raw, obs_sequences))
    epoch_nb = 0
    mean_elbos = [old_mean_epoch_elbo - average_evidence_dataset]
    # while eps > 1e-2:
    for _ in range(10):
        epoch_elbo = 0.0
        for batch in obs_sequences: 
            p_raw, q_raw, opt_state, elbo_value = step(p_raw, q_raw, opt_state, batch)
            epoch_elbo += elbo_value
        mean_epoch_elbo = epoch_elbo/len(obs_sequences)
        
        eps = jnp.abs(mean_epoch_elbo - old_mean_epoch_elbo)
        epoch_nb+=1
        mean_elbos.append(mean_epoch_elbo - average_evidence_dataset)
        old_mean_epoch_elbo = mean_epoch_elbo
    return q_raw, mean_elbos

fitted_q_raw, mean_elbos = fit(p_raw, q_raw, optimizer)

plt.plot(mean_elbos)
plt.xlabel('Epoch nb'), 
plt.ylabel('$|\mathcal{L}(\\theta,\\phi)- \log p_\\theta(x)|$')
plt.show()

fitted_q = parameters_from_raw_parameters(fitted_q_raw)

#%% 