from jax.scipy.special import logsumexp
import jax.numpy as jnp
from jax import vmap, jit, lax, random
from jax import lax
from functools import partial

from backward_ica.utils import tree_prepend

def exp_and_normalize(x):

    x = jnp.exp(x - x.max())
    return x / x.sum()

def compute_pred_likel(log_probs):
    return logsumexp(log_probs)


class SMC:

    def __init__(self, transition_kernel, emission_kernel, prior_dist):

        self.transition_kernel = transition_kernel 
        self.emission_kernel = emission_kernel
        self.prior_sampler = prior_dist.sample 
    
    
    def init(self, prior_key, obs, prior_params, emission_params, num_particles):

        particles = vmap(self.prior_sampler, in_axes=(0,None))(random.split(prior_key, num_particles), prior_params)
        log_probs = self.update(particles, obs, emission_params)
        return log_probs, particles


    def predict(self, resampling_key, proposal_key, log_probs, particles, transition_params):

        particles = self.resample(resampling_key, log_probs, particles)

        particles = vmap(self.transition_kernel.sample, in_axes=(0,0,None))(random.split(proposal_key, len(particles)), particles, transition_params)

        return particles

    def update(self, particles, obs, emission_params):

        log_probs = vmap(self.emission_kernel.logpdf, in_axes=(None, 0, None))(obs, 
                                                                        particles, 
                                                                        emission_params)
        return log_probs


    def resample(key, log_probs, particles):

        return random.choice(key=key, a=particles, p=exp_and_normalize(log_probs), replace=True, shape=(len(particles),))

    def filter_seq(self, key, obs_seq, params, num_particles):

        prior_key, proposal_key, resampling_key = random.split(key,3)
        
        init_log_probs, init_particles = self.init(prior_key, obs_seq[0], params.prior, params.emission, num_particles)
        likel = compute_pred_likel(init_log_probs)

        @jit 
        def _filter_step(carry, x):
            log_probs, particles, likel = carry
            obs, proposal_key, resampling_key = x
            particles = self.predict(resampling_key, proposal_key, log_probs, particles, params.transition)
            log_probs = self.update(particles, obs, params.emission)
            likel += compute_pred_likel(log_probs)
            return (log_probs, particles, likel), None

        proposal_keys = random.split(proposal_key, len(obs_seq) - 1)
        resampling_keys = random.split(resampling_key, len(obs_seq) - 1)

        terminal_log_probs, terminal_particles, likel = lax.scan(_filter_step, 
                                                    init=(init_log_probs, init_particles, likel), 
                                                    xs=(obs_seq[1:], proposal_keys, resampling_keys))[0]

        return terminal_log_probs, terminal_particles, likel - len(obs_seq)*jnp.log(num_particles)


    def compute_filt_seq(self, key, obs_seq, params, num_particles):


        prior_key, proposal_key, resampling_key = random.split(key,3)
        
        init_log_probs, init_particles = self.init(prior_key, obs_seq[0], params.prior, params.emission, num_particles)

        @jit 
        def _filter_step(carry, x):
            log_probs, particles = carry
            obs, proposal_key, resampling_key = x
            particles = self.predict(resampling_key, proposal_key, log_probs, particles, params.transition)
            log_probs = self.update(particles, obs, params.emission)
            return (log_probs, particles), (log_probs, particles)

        proposal_keys = random.split(proposal_key, len(obs_seq) - 1)
        resampling_keys = random.split(resampling_key, len(obs_seq) - 1)

        log_probs, particles = lax.scan(_filter_step, 
                                        init=(init_log_probs, init_particles), 
                                        xs=(obs_seq[1:], proposal_keys, resampling_keys))[1]

        return tree_prepend(init_log_probs, log_probs), tree_prepend(init_particles, particles)

    def smooth_from_filt_seq(self, key, filt_seq, params):

        log_probs_seq, particles_seq = filt_seq

        @jit
        def _sample_path(key, log_probs_seq, particles_seq):

            path_keys = random.split(key, len(particles_seq))

            last_sample = random.choice(path_keys[-1], a=particles_seq[-1], p=exp_and_normalize(log_probs_seq[-1]))

            def _step(carry, x):
                next_sample = carry 
                log_probs, particles, key = x 
                log_probs_backwd = log_probs + vmap(self.transition_kernel.logpdf, in_axes=(None, 0, None))(next_sample, particles, params.transition)
                sample = random.choice(key, a=particles, p=exp_and_normalize(log_probs_backwd))
                return sample, sample

            samples = lax.scan(_step, init=last_sample, xs=(log_probs_seq[:-1], particles_seq[:-1], path_keys[:-1]), reverse=True)[1]
            
            return jnp.concatenate((samples, last_sample[None,:]))

        keys = random.split(key, len(particles_seq[0]))

        paths = vmap(_sample_path, in_axes=(0,None,None))(keys, log_probs_seq, particles_seq)

        return jnp.mean(paths, axis=0), jnp.var(paths, axis=0)
        

