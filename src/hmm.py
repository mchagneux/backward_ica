from .misc import * 
from jax.random import multivariate_normal, uniform
import jax.numpy as jnp
from jax import random
from jax import lax

class LinearGaussianHMM:

    def get_random_model(key, state_dim=2, obs_dim=2):

        key, *subkeys = random.split(key, 3)
        prior_mean = uniform(subkeys[0], shape=(state_dim,))
        prior_cov = jnp.diag(uniform(subkeys[1], shape=(state_dim,))**2)

        key, *subkeys = random.split(key, 4)
        transition_weight = jnp.diag(uniform(subkeys[0], shape=(state_dim,)))
        transition_bias = uniform(subkeys[1], shape=(state_dim,))
        transition_cov = jnp.diag(uniform(subkeys[2], shape=(state_dim,))**2)

        key, *subkeys = random.split(key, 4)
        emission_weight = uniform(subkeys[0], shape=(state_dim, obs_dim))
        emission_bias = uniform(subkeys[1], shape=(obs_dim,))
        emission_cov = jnp.diag(uniform(subkeys[2], shape=(obs_dim,)) ** 2)

        return key, Model(prior=Prior(prior_mean, prior_cov),
                    transition=Transition(transition_weight, transition_bias, transition_cov),
                    emission=Emission(emission_weight, emission_bias, emission_cov))

    def sample_single_step_joint(carry, x):
        key, previous_state, transition, emission = carry
        key, *subkeys = random.split(key, 3)

        new_state = multivariate_normal(key=subkeys[0], 
                                        mean=transition.weight @ previous_state + transition.bias,
                                        cov=transition.cov)

        new_obs = multivariate_normal(key=subkeys[1], 
                                    mean=emission.weight @ new_state + emission.bias,
                                    cov=transition.cov)

        return (key, new_state, transition, emission), (new_state, new_obs)

    def sample_joint_sequence(key, model:Model, length):

        key, *subkeys = random.split(key, 3)
        init_state = multivariate_normal(key=subkeys[0], 
                                        mean=model.prior.mean,
                                        cov=model.prior.cov)

        init_obs = multivariate_normal(key=subkeys[1], 
                                        mean=model.emission.weight @ init_state + model.emission.bias, 
                                        cov=model.emission.cov)

        (key, *_), (state_samples, obs_samples) = lax.scan(f=LinearGaussianHMM.sample_single_step_joint, 
                                                init=(key, init_state, model.transition, model.emission), 
                                                length=length-1,
                                                xs=None)
        
        state_samples = jnp.concatenate((init_state[None,:], state_samples))
        obs_samples = jnp.concatenate((init_obs[None,:], obs_samples))

        return key, state_samples, obs_samples


        

        

