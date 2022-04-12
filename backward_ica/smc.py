from jax.scipy.special import logsumexp
from jax.random import multivariate_normal, choice
import jax.numpy as jnp
from jax import vmap, jit
from jax import lax
from .hmm import GaussianKernel


def exp_and_normalize(x):

    x = x - x.max()
    return jnp.exp(x) / x.sum()

def smc_init(key, obs, prior_sampler, emission_kernel, prior_params, emission_params, num_particles):

    particles = prior_sampler(key, prior_params, (num_particles,))

    return smc_update(particles, obs, emission_kernel, emission_params)

def smc_predict(keys, weights, particles, transition_kernel:GaussianKernel, transition_params, num_particles):

    key_resampling, keys_transition = keys

    particles = resample(key_resampling, weights, particles, num_particles)

    particles = vmap(transition_kernel.sample, in_axes=(0,0,None))(keys_transition, particles, transition_params)

    return particles

def smc_update(particles, obs, emission_kernel:GaussianKernel, emission_params):

    log_probs = vmap(emission_kernel.logpdf, in_axes=(None, 0, None))(obs, particles, emission_params)

    weights = exp_and_normalize(log_probs)

    likel_pred = None 

    return weights, likel_pred

def resample(key, weights, particles, num_particles):

    return choice(key, particles, weights, replace=True, shape=(num_particles,))


def smc_filter_seq(key, obs_seq, prior_sampler, transition_kernel, emission_kernel, params, num_particles):

    init_weights, init_particles, likel = smc_init(key, obs_seq[0], prior_sampler, emission_kernel, params.prior, params.emission, num_particles)

    def _filter_step(carry, x):
        weights, particles, likel = carry 
        obs = x
        particles = smc_predict(key, weights, particles, transition_kernel, params.transition, num_particles)
        weights, likel_pred = smc_update(particles, obs, emission_kernel, params.emission)
        return (weights, particles, likel + likel_pred), (weights, particles)

    (terminal_weights, terminal_particles, likel), (all_weights, all_particles) = lax.scan(_filter_step, init=(init_weights, init_particles, likel), xs=key)

    return terminal_weights, terminal_particles, likel

def smc_smoother_seq(key):

    pass 


