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



state_dim, obs_dim = 2, 2 
num_sequences = 20
length = 8

key, *subkeys = random.split(key,3)

p_raw = LinearGaussianHMM.get_random_model(key=subkeys[0], state_dim=state_dim, obs_dim=obs_dim)
q_raw = LinearGaussianHMM.get_random_model(key=subkeys[1], state_dim=state_dim, obs_dim=obs_dim)

p = actual_model_from_raw_parameters(p_raw)
q = actual_model_from_raw_parameters(q_raw)

linear_gaussian_sampler = jax.vmap(LinearGaussianHMM.sample_joint_sequence, in_axes=(0, None, None))
key, *subkeys = random.split(key, num_sequences+1)
state_sequences, obs_sequences = linear_gaussian_sampler(jnp.array(subkeys), p, length)


filter_obs_sequences = jax.vmap(kalman.filter, in_axes=(0, None))
elbo_sequences = jax.vmap(linear_gaussian_elbo, in_axes=(None, None, 0))

average_evidence_across_sequences = jnp.mean(filter_obs_sequences(obs_sequences, p)[-1])
average_elbo_across_sequences_with_true_model = jnp.mean(elbo_sequences(p_raw, p_raw, obs_sequences))
print('Difference mean evidence Kalman and mean ELBO when q=p:', jnp.abs(average_evidence_across_sequences-average_elbo_across_sequences_with_true_model))
average_elbo_across_sequences_with_init_q = jnp.mean(elbo_sequences(p_raw, q_raw, obs_sequences))
print('Different mean evidence and mean ELBO when q=q0:', jnp.abs(average_evidence_across_sequences-average_elbo_across_sequences_with_init_q))


optimizer = optax.adam(learning_rate=-1e-3)

loss = lambda p_raw, q_raw, observations: linear_gaussian_elbo(p_raw, q_raw, observations)

def fit(p_raw, q_raw, optimizer: optax.GradientTransformation) -> optax.Params:
    opt_state = optimizer.init(q_raw)

    @jax.jit
    def step(p_raw, q_raw, opt_state, batch):
        loss_value, grads = jax.value_and_grad(loss, argnums=1)(p_raw, q_raw, batch)
        updates, opt_state = optimizer.update(grads, opt_state, q_raw)
        q_raw = optax.apply_updates(q_raw, updates)
        return p_raw, q_raw, opt_state, -loss_value

    eps = jnp.inf
    old_mean_epoch_loss = -average_elbo_across_sequences_with_init_q
    epoch_nb = 0
    while eps > 1e-2:
        epoch_loss = 0.0
        for batch in obs_sequences: 
            p_raw, q_raw, opt_state, loss_value = step(p_raw, q_raw, opt_state, batch)
            epoch_loss += loss_value
        mean_epoch_loss = epoch_loss/len(obs_sequences)
        eps = jnp.abs(mean_epoch_loss - old_mean_epoch_loss)
        if epoch_nb % 10 == 0: print('Change in average elbo from last iteration:', eps)
        epoch_nb+=1
        old_mean_epoch_loss = mean_epoch_loss
    return q_raw 


fitted_q_raw = fit(p_raw, q_raw, optimizer)








