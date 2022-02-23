from abc import abstractmethod, ABCMeta
from src.misc import QuadForm
from src.elbo import linear_gaussian_elbo
from src.hmm import LinearGaussianHMM
from src import kalman
from jax.random import PRNGKey
import jax.numpy as jnp
import jax
import optax
from jax import random

key = PRNGKey(0)
key, subkey = random.split(key)
p = LinearGaussianHMM.get_random_model(key=subkey, state_dim=2, obs_dim=2)

num_sequences = 10 
length = 8

linear_gaussian_sampler = jax.vmap(LinearGaussianHMM.sample_joint_sequence, in_axes=(0, None, None))
key, *subkeys = random.split(key, num_sequences+1)
state_sequences, obs_sequences = linear_gaussian_sampler(jnp.array(subkeys), p, length)


filter_obs_sequences = jax.vmap(kalman.filter, in_axes=(0, None))
mean_elbo_across_sequences = jax.vmap(linear_gaussian_elbo, in_axes=(None, None,0))
 
average_evidence_across_sequences = jnp.mean(filter_obs_sequences(obs_sequences, p)[-1])

print('Average evidence across sequences:', jnp.mean(mean_elbo_across_sequences(p, p, obs_sequences)))
print('Average elbo across sequences', jnp.mean(mean_elbo_across_sequences(p, p, obs_sequences)))

grad_elbo_theta = jax.grad(linear_gaussian_elbo, 
                        argnums=1)

grads_theta = grad_elbo_theta(p,p,obs_sequences[0])

# elbo = linear_gaussian_elbo(p=p, 
#                             q=p, 
#                             observations=observations)

# print(jnp.abs(elbo - likelihood))




