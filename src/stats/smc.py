from jax.scipy.special import logsumexp


from backward_ica.utils import *

def compute_pred_likel(probs):
    return 


class SMC:

    def __init__(self, transition_kernel, emission_kernel, prior_dist, num_particles=1000, num_smooth_particles=None):

        self.transition_kernel = transition_kernel 
        self.emission_kernel = emission_kernel
        self.prior_sampler = prior_dist.sample 
        self.num_filt_particles = num_particles
        if num_smooth_particles is None:
            self.num_smooth_particles = num_particles
        else: self.num_smooth_particles = num_smooth_particles
    
    
    def init(self, prior_key, obs, prior_params, emission_params):

        particles = vmap(self.prior_sampler, in_axes=(0,None))(random.split(prior_key, self.num_filt_particles), prior_params)
        probs, likel = self.update(particles, obs, emission_params)
        return probs, particles, likel


    def resample(self, key, probs, particles):

        return random.choice(key=key, a=particles, p=probs, replace=True, shape=(self.num_filt_particles,))


    def predict(self, resampling_key, proposal_key, probs, particles, transition_params):

        particles = self.resample(resampling_key, probs, particles)

        particles = vmap(self.transition_kernel.sample, in_axes=(0,0,None))(random.split(proposal_key, self.num_filt_particles), particles, transition_params)

        return particles

    def update(self, particles, obs, emission_params):

        log_probs = vmap(self.emission_kernel.logpdf, in_axes=(None, 0, None))(obs, 
                                                                        particles, 
                                                                        emission_params)
        return exp_and_normalize(log_probs), logsumexp(log_probs)


    def compute_filt_params_seq(self, key, obs_seq, params):


        prior_key, proposal_key, resampling_key = random.split(key,3)
        
        init_probs, init_particles, init_likel = self.init(prior_key, obs_seq[0], params.prior, params.emission)

        @jit 
        def _filter_step(carry, x):
            probs, particles = carry
            obs, proposal_key, resampling_key = x
            particles = self.predict(resampling_key, proposal_key, probs, particles, params.transition)
            probs, likel = self.update(particles, obs, params.emission)

            return (probs, particles), (probs, particles, likel)

        proposal_keys = random.split(proposal_key, len(obs_seq) - 1)
        resampling_keys = random.split(resampling_key, len(obs_seq) - 1)

        probs, particles, likel = lax.scan(_filter_step, 
                                        init=(init_probs, init_particles), 
                                        xs=(obs_seq[1:], proposal_keys, resampling_keys))[1]

        return (tree_prepend(init_probs, probs), tree_prepend(init_particles, particles)), jnp.sum(likel) + init_likel - len(obs_seq)*jnp.log(self.num_filt_particles)

    def smooth_from_filt_seq(self, key, filt_seq, params):

        probs_seq, particles_seq = filt_seq

        @jit
        def _sample_path(key, probs_seq, particles_seq):

            path_keys = random.split(key, len(particles_seq))

            last_sample = random.choice(path_keys[-1], a=particles_seq[-1], p=probs_seq[-1])

            def _step(carry, x):
                next_sample = carry 
                probs, particles, key = x 
                log_probs_backwd = jnp.log(probs) + vmap(self.transition_kernel.logpdf, in_axes=(None, 0, None))(next_sample, particles, params.transition)
                sample = random.choice(key, a=particles, p=exp_and_normalize(log_probs_backwd))
                return sample, sample

            samples = lax.scan(_step, init=last_sample, xs=(probs_seq[:-1], particles_seq[:-1], path_keys[:-1]), reverse=True)[1]
            
            return jnp.concatenate((samples, last_sample[None,:]))

        keys = random.split(key, self.num_smooth_particles)

        paths = vmap(_sample_path, in_axes=(0,None,None))(keys, probs_seq, particles_seq)

        return jnp.transpose(paths, axes=(1,0,2))
        

    # def smooth_from_filt_seq(self, key, filt_seq, params):

    #     probs_seq, particles_seq = filt_seq

    #     sigma_plus = 1 / jnp.sqrt(((2*jnp.pi)**self.transition_kernel.in_dim)*jnp.linalg.det(params.transition.noise.scale.cov))

    #     @jit
    #     def _sample_path(key, probs_seq, particles_seq):

    #         path_keys = random.split(key, len(particles_seq))

    #         last_sample = random.choice(path_keys[-1], a=particles_seq[-1], p=probs_seq[-1])
        

    #         def _step(carry, x):
    #             next_sample = carry  
    #             key, probs, particles = x

    #             key, key_sample, key_unif = random.split(key, 3)
    #             sample = random.choice(key_sample, a=particles, p=probs)
    #             unif_draw = random.uniform(key_unif, minval=0, maxval=1)
    #             val = sample, next_sample, unif_draw, key, probs, particles

    #             def _cond_fun(val):
    #                 sample, next_sample, unif_draw, _, _, _ = val
    #                 test = self.transition_kernel.pdf(next_sample, sample, params.transition) / sigma_plus
    #                 return unif_draw > test 

    #             def _while_body(val):
    #                 sample, next_sample, unif_draw, key, probs, particles = val
    #                 key, key_sample, key_unif = random.split(key, 3)
    #                 sample = random.choice(key_sample, a=particles, p=probs)
    #                 unif_draw = random.uniform(key_unif, minval=0, maxval=1)
    #                 val = sample, next_sample, unif_draw, key, probs, particles
    #                 return val 
                
    #             sample = lax.while_loop(_cond_fun, _while_body, init_val=val)[0]

    #             return sample, sample
    
    #         samples = lax.scan(_step, init=last_sample, xs=(path_keys[:-1], probs_seq[:-1], particles_seq[:-1]), reverse=True)[1]
            
    #         return jnp.concatenate((samples, last_sample[None,:]))

    #     keys = random.split(key, self.num_particles)

    #     paths = vmap(_sample_path, in_axes=(0,None,None))(keys, probs_seq, particles_seq)

    #     return jnp.transpose(paths, axes=(1,0,2))

            
        
