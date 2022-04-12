from jax.scipy.special import logsumexp
import jax.numpy as jnp
from jax import vmap, jit, lax, random
from jax import lax

def exp_and_normalize(x):

    x = x - x.max()
    return jnp.exp(x) / x.sum()

def smc_init(prior_keys, obs, prior_sampler, emission_kernel, prior_params, emission_params, num_particles):

    particles = vmap(prior_sampler, in_axes=(0,None))(prior_keys, prior_params)
    log_probs, likel = smc_update(particles, obs, emission_kernel, emission_params, num_particles)
    return log_probs, particles, likel

def smc_predict(resampling_key, proposal_key, log_probs, particles, transition_kernel, transition_params, num_particles):

    particles = resample(resampling_key, log_probs, particles, num_particles)

    mapped_particles = transition_kernel.map(particles, transition_params)
    particles = transition_kernel.sample(proposal_key, mapped_particles, transition_params)

    return particles

def smc_update(particles, obs, emission_kernel, emission_params, num_particles):

    log_probs = vmap(emission_kernel.logpdf, in_axes=(None, 0, None))(obs, 
                                                                    emission_kernel.map(particles, emission_params), 
                                                                    emission_params)

    likel_pred = - jnp.log(num_particles) + logsumexp(log_probs)

    return log_probs, likel_pred

def resample(key, log_probs, particles, num_particles):

    return random.choice(key=key, a=particles, p=exp_and_normalize(log_probs), replace=True, shape=(num_particles,))

def smc_filter_seq(prior_keys, resampling_keys, proposal_keys, obs_seq, prior_sampler, transition_kernel, emission_kernel, params, num_particles):

    init_log_probs, init_particles, likel = smc_init(prior_keys, obs_seq[0], prior_sampler, emission_kernel, params.prior, params.emission, num_particles)

    @jit
    def _filter_step(carry, x):
        log_probs, particles, likel = carry
        obs, resampling_key, proposal_key = x
        particles = smc_predict(resampling_key, proposal_key, log_probs, particles, transition_kernel, params.transition, num_particles)
        log_probs, likel_pred = smc_update(particles, obs, emission_kernel, params.emission, num_particles)
        return (log_probs, particles, likel + likel_pred), (log_probs, particles)

    (terminal_log_probs, terminal_particles, likel), (all_log_probs, all_particles) = lax.scan(_filter_step, 
                                                                                        init=(init_log_probs, init_particles, likel), 
                                                                                        xs=(obs_seq[1:], resampling_keys, proposal_keys))

    return terminal_log_probs, terminal_particles, likel


