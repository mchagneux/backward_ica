from jax.scipy.special import logsumexp
import jax.numpy as jnp
from jax import vmap, jit, lax, random
from jax import lax
from functools import partial

from backward_ica.utils import tree_prepend

def exp_and_normalize(x):

    x = jnp.exp(x - x.max())
    return x / x.sum()

def smc_init(prior_key, obs, prior_sampler, emission_kernel, prior_params, emission_params, num_particles):

    particles = vmap(prior_sampler, in_axes=(0,None))(random.split(prior_key, num_particles), prior_params)
    log_probs = smc_update(particles, obs, emission_kernel, emission_params)
    return log_probs, particles


def smc_predict(resampling_key, proposal_key, log_probs, particles, transition_kernel, transition_params):

    particles = resample(resampling_key, log_probs, particles)

    particles = vmap(transition_kernel.sample, in_axes=(0,0,None))(random.split(proposal_key, len(particles)), particles, transition_params)

    return particles

def smc_update(particles, obs, emission_kernel, emission_params):

    log_probs = vmap(emission_kernel.logpdf, in_axes=(None, 0, None))(obs, 
                                                                    particles, 
                                                                    emission_params)
    return log_probs

def compute_pred_likel(log_probs):
    return logsumexp(log_probs)

def resample(key, log_probs, particles):

    return random.choice(key=key, a=particles, p=exp_and_normalize(log_probs), replace=True, shape=(len(particles),))

def smc_filter_seq(key, obs_seq, params, prior_sampler, transition_kernel, emission_kernel, num_particles):

    prior_key, proposal_key, resampling_key = random.split(key,3)
    
    init_log_probs, init_particles = smc_init(prior_key, obs_seq[0], prior_sampler, emission_kernel, params.prior, params.emission, num_particles)
    likel = compute_pred_likel(init_log_probs)

    @jit 
    def _filter_step(carry, x):
        log_probs, particles, likel = carry
        obs, proposal_key, resampling_key = x
        particles = smc_predict(resampling_key, proposal_key, log_probs, particles, transition_kernel, params.transition)
        log_probs = smc_update(particles, obs, emission_kernel, params.emission)
        likel += compute_pred_likel(log_probs)
        return (log_probs, particles, likel), None

    proposal_keys = random.split(proposal_key, len(obs_seq) - 1)
    resampling_keys = random.split(resampling_key, len(obs_seq) - 1)

    terminal_log_probs, terminal_particles, likel = lax.scan(_filter_step, 
                                                init=(init_log_probs, init_particles, likel), 
                                                xs=(obs_seq[1:], proposal_keys, resampling_keys))[0]

    return terminal_log_probs, terminal_particles, likel - len(obs_seq)*jnp.log(num_particles)


def smc_smooth_additive_func(obs_seq, params, h_tilde, prior_keys, resampling_keys, proposal_keys, prior_sampler, transition_kernel, emission_kernel, num_particles):

    init_log_probs, init_particles = smc_init(prior_keys, obs_seq[0], prior_sampler, emission_kernel, params.prior, params.emission)
    init_tau = jnp.zeros((num_particles,))

    def _smoothing_step(carry, x):
        prev_log_probs, prev_particles, prev_tau = carry 
        obs, resampling_key, proposal_key = x
        mapped_prev_particles = transition_kernel.map(prev_particles, params.transition)
        particles = smc_predict(resampling_key, proposal_key, prev_log_probs, prev_particles, transition_kernel, params.transition, num_particles)

        def new_tau_component(particle):
            log_probs_backward = prev_log_probs + vmap(lambda mapped_prev_particle: transition_kernel.logpdf(particle, mapped_prev_particle, params.transition))(mapped_prev_particles)
            normalized_weights = exp_and_normalize(log_probs_backward)
            sum_component = lambda normalized_weight, tau_component, prev_particle: normalized_weight * (tau_component + h_tilde(prev_particle, particle))
            return jnp.sum(vmap(sum_component)(normalized_weights, prev_tau, prev_particles))
            
        tau = vmap(new_tau_component)(particles)
        log_probs = smc_update(particles, obs, emission_kernel, params.emission)
        
        return (log_probs, particles, tau), jnp.sum(exp_and_normalize(log_probs) * tau)

    smoothing_seq = lax.scan(_smoothing_step, 
                            init=(init_log_probs, init_particles, init_tau), 
                            xs=(obs_seq[1:], resampling_keys, proposal_keys))[1]
    
    smoothing_seq = jnp.concatenate((jnp.zeros((1,)), smoothing_seq))

    
    return smoothing_seq


def smc_compute_filt_seq(key, obs_seq, params, prior_sampler, transition_kernel, emission_kernel, num_particles):


    prior_key, proposal_key, resampling_key = random.split(key,3)
    
    init_log_probs, init_particles = smc_init(prior_key, obs_seq[0], prior_sampler, emission_kernel, params.prior, params.emission, num_particles)

    @jit 
    def _filter_step(carry, x):
        log_probs, particles = carry
        obs, proposal_key, resampling_key = x
        particles = smc_predict(resampling_key, proposal_key, log_probs, particles, transition_kernel, params.transition)
        log_probs = smc_update(particles, obs, emission_kernel, params.emission)
        return (log_probs, particles), (log_probs, particles)

    proposal_keys = random.split(proposal_key, len(obs_seq) - 1)
    resampling_keys = random.split(resampling_key, len(obs_seq) - 1)

    log_probs, particles = lax.scan(_filter_step, 
                                    init=(init_log_probs, init_particles), 
                                    xs=(obs_seq[1:], proposal_keys, resampling_keys))[1]

    return tree_prepend(init_log_probs, log_probs), tree_prepend(init_particles, particles)

def smc_smooth_from_filt_seq(key, filt_seq, params, transition_kernel):

    log_probs_seq, particles_seq = filt_seq

    @jit
    def _sample_path(key, log_probs_seq, particles_seq):

        path_keys = random.split(key, len(particles_seq))

        last_sample = random.choice(path_keys[-1], a=particles_seq[-1], p=exp_and_normalize(log_probs_seq[-1]))

        def _step(carry, x):
            next_sample = carry 
            log_probs, particles, key = x 
            log_probs_backwd = log_probs + vmap(transition_kernel.logpdf, in_axes=(None, 0, None))(next_sample, particles, params.transition)
            sample = random.choice(key, a=particles, p=exp_and_normalize(log_probs_backwd))
            return sample, sample

        samples = lax.scan(_step, init=last_sample, xs=(log_probs_seq[:-1], particles_seq[:-1], path_keys[:-1]), reverse=True)[1]
        
        return jnp.concatenate((samples, last_sample[None,:]))

    keys = random.split(key, len(particles_seq[0]))

    paths = vmap(_sample_path, in_axes=(0,None,None))(keys, log_probs_seq, particles_seq)

    return jnp.mean(paths, axis=0), jnp.var(paths, axis=0)
    

