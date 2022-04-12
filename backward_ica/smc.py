from jax.scipy.special import logsumexp
from jax.random import multivariate_normal
import jax.numpy as jnp
from jax import vmap, jit
from jax import lax
from .hmm import GaussianKernel

def smc_init(keys, prior_sampler, emission_kernel, params, num_particles):

    key_prior, key_update = keys
    particles = prior_sampler(key_prior, params.prior, (num_particles,))

    return smc_update(key_update, particles, emission_kernel, params)

def smc_predict(keys, filt_state, transition_kernel:GaussianKernel, params):

    key_resampling, key_transition = keys

    particles = resample(key_resampling, filt_state)

    particles = vmap(transition_kernel.sample, in_axes=(0,0,None))(key_transition, particles, params.transition)

    return particles

def smc_update(key, pred_state, emission_kernel, params):

    pass 

def resample(key, state):
    pass 


def smc_filter_seq(key):

    pass


def smc_smoother_seq(key):

    pass 


