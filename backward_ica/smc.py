from jax.scipy.special import logsumexp
import jax.numpy as jnp
from jax import vmap, jit, lax, random
from jax import lax
def exp_and_normalize(x):

    x = jnp.exp(x - x.max())
    return x / x.sum()

def smc_init(prior_keys, obs, prior_sampler, emission_kernel, prior_params, emission_params):

    particles = vmap(prior_sampler, in_axes=(0,None))(prior_keys, prior_params)
    log_probs = smc_update(particles, obs, emission_kernel, emission_params)
    return log_probs, particles

def smc_predict(resampling_key, proposal_key, log_probs, particles, transition_kernel, transition_params, num_particles):

    particles = resample(resampling_key, log_probs, particles, num_particles)

    mapped_particles = transition_kernel.map(particles, transition_params)
    particles = transition_kernel.sample(proposal_key, mapped_particles, transition_params)

    return particles

def smc_update(particles, obs, emission_kernel, emission_params):

    log_probs = vmap(emission_kernel.logpdf, in_axes=(None, 0, None))(obs, 
                                                                    emission_kernel.map(particles, emission_params), 
                                                                    emission_params)
    return log_probs

def compute_pred_likel(log_probs, num_particles):
    return -jnp.log(num_particles) + logsumexp(log_probs)

def resample(key, log_probs, particles, num_particles):

    return random.choice(key=key, a=particles, p=exp_and_normalize(log_probs), replace=True, shape=(num_particles,))

def smc_filter_seq(prior_keys, resampling_keys, proposal_keys, obs_seq, prior_sampler, transition_kernel, emission_kernel, params, num_particles):

    init_log_probs, init_particles = smc_init(prior_keys, obs_seq[0], prior_sampler, emission_kernel, params.prior, params.emission)
    likel = compute_pred_likel(init_log_probs, num_particles)

    @jit
    def _filter_step(carry, x):
        log_probs, particles, likel = carry
        obs, resampling_key, proposal_key = x
        particles = smc_predict(resampling_key, proposal_key, log_probs, particles, transition_kernel, params.transition, num_particles)
        log_probs = smc_update(particles, obs, emission_kernel, params.emission)
        return (log_probs, particles, likel + compute_pred_likel(log_probs, num_particles)), None

    terminal_log_probs, terminal_particles, likel = lax.scan(_filter_step, 
                                                init=(init_log_probs, init_particles, likel), 
                                                xs=(obs_seq[1:], resampling_keys, proposal_keys))[0]

    return terminal_log_probs, terminal_particles, likel


def smc_smooth_seq(obs_seq, params, h_tilde, prior_keys, resampling_keys, proposal_keys, prior_sampler, transition_kernel, emission_kernel, num_particles):

    init_log_probs, init_particles = smc_init(prior_keys, obs_seq[0], prior_sampler, emission_kernel, params.prior, params.emission)
    init_tau = jnp.zeros((num_particles,))

    @jit
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
