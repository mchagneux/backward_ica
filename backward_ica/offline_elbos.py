import jax
from jax import vmap, lax, numpy as jnp
from .stats.hmm import *
from .utils import *
from backward_ica.stats import BackwardSmoother, TwoFilterSmoother


class GeneralForwardELBO:

    def __init__(self, p:HMM, q:TwoFilterSmoother, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples 

    def __call__(self, key, obs_seq, compute_up_to, theta, phi):

        theta.compute_covs()
        def _monte_carlo_sample(key, obs_seq, init_state, backwd_variables_seq):
            
            def _sample_step(prev_sample, x):

                key, obs, idx = x

                def false_fun(key, prev_sample, obs, idx):
                    return prev_sample, 0.0

                def true_fun(key, prev_sample, obs, idx):

                    def init_term(key, prev_sample, obs, idx):
                        init_law_params = self.q.compute_marginal(self.q.init_filt_params(init_state, phi), 
                                                tree_get_idx(0, backwd_variables_seq)) 
                        init_sample = self.q.marginal_dist.sample(key, init_law_params)
                        init_term = self.p.emission_kernel.logpdf(obs, init_sample, theta.emission) \
                                    + self.p.prior_dist.logpdf(init_sample, theta.prior) \
                                    - self.q.marginal_dist.logpdf(init_sample, init_law_params)
                        return init_sample, init_term


                    def other_terms(key, prev_sample, obs, idx):
                        forward_params = self.q.forward_params_from_backwd_var(tree_get_idx(idx, backwd_variables_seq), phi)
                        sample = self.q.forward_kernel.sample(key, prev_sample, forward_params)
                        emission_term_p = self.p.emission_kernel.logpdf(obs, sample, theta.emission)
                        transition_term_p = self.p.transition_kernel.logpdf(sample, prev_sample, theta.transition)
                        fwd_term_q = -self.q.forward_kernel.logpdf(sample, prev_sample, forward_params)
                        return sample,  emission_term_p + transition_term_p + fwd_term_q


                    return lax.cond(idx > 0, other_terms, init_term, key, prev_sample, obs, idx)

                return lax.cond(idx <= compute_up_to, true_fun, false_fun, key, prev_sample, obs, idx)

            terms = lax.scan(f=_sample_step, 
                            init=jnp.empty((self.p.state_dim,)), 
                            xs=(jax.random.split(key, obs_seq.shape[0]), 
                                obs_seq, 
                                jnp.arange(0, len(obs_seq))), 
                                reverse=False)[1]


            return jnp.sum(terms)


        parallel_sampler = vmap(_monte_carlo_sample, in_axes=(0,None,None,None))

        keys = jax.random.split(key, self.num_samples)

        state_seq = self.q.compute_state_seq(obs_seq, phi)
        backwd_variables_seq = self.q.compute_backwd_variables_seq(state_seq, compute_up_to, phi)


        mc_samples = parallel_sampler(keys, 
                                    obs_seq, 
                                    tree_get_idx(0, state_seq),
                                    backwd_variables_seq)

        return jnp.mean(mc_samples) / (compute_up_to + 1)

class GeneralBackwardELBO:

    def __init__(self, p:HMM, q:BackwardSmoother, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples

    def __call__(self, key, obs_seq, compute_up_to, theta:HMM.Params, phi):

        theta.compute_covs()

        def _monte_carlo_sample(key, obs_seq, state_seq):

            def _sample_step(next_sample, x):
                
                key, obs, idx = x

                def false_fun(key, next_sample, obs, idx):
                    return next_sample, 0.0

                def true_fun(key, next_sample, obs, idx):

                    def last_term(key, next_sample, obs, idx):
                        filt_params = self.q.filt_params_from_state(tree_get_idx(idx, state_seq), 
                                                                    phi)
                        sample = self.q.filt_dist.sample(key, filt_params)
                        term = -self.q.filt_dist.logpdf(sample, filt_params) \
                                + self.p.emission_kernel.logpdf(obs, sample, theta.emission)
                        return sample, term

                    def other_terms(key, next_sample, obs, idx):
                        
                        backwd_params = self.q.backwd_params_from_state(tree_get_idx(idx, state_seq), phi)

                        sample = self.q.backwd_kernel.sample(key, next_sample, backwd_params)

                        emission_term_p = self.p.emission_kernel.logpdf(obs, sample, theta.emission)

                        transition_term_p = self.p.transition_kernel.logpdf(next_sample, sample, theta.transition)

                        backwd_term_q = -self.q.backwd_kernel.logpdf(sample, next_sample, backwd_params)

                        term = emission_term_p + transition_term_p + backwd_term_q

                        return sample, term 

                    return lax.cond(idx < compute_up_to, other_terms, last_term, key, next_sample, obs, idx)
                    
                return lax.cond(idx <= compute_up_to, true_fun, false_fun, key, next_sample, obs, idx)
            
            init_sample, terms = lax.scan(_sample_step, 
                                        init=jnp.empty((self.p.state_dim,)), 
                                        xs=(jax.random.split(key, obs_seq.shape[0]), 
                                            obs_seq, 
                                            jnp.arange(0, len(obs_seq))),
                                        reverse=True)

            return self.p.prior_dist.logpdf(init_sample, theta.prior) + jnp.sum(terms)

        parallel_sampler = vmap(_monte_carlo_sample, in_axes=(0,None,None))

        state_seq = self.q.compute_state_seq(obs_seq, compute_up_to, phi)

        mc_samples = parallel_sampler(jax.random.split(key, self.num_samples),
                                    obs_seq, 
                                    state_seq)

        return jnp.mean(mc_samples) / (compute_up_to + 1)

# class OfflineAdditiveSmoothing:

#     def __init__(self, p:HMM, q:BackwardSmoother, functional, num_samples):

#         self.p = p 
#         self.q = q 
#         self.num_samples = num_samples 
#         self.functional = functional


#     def __call__(self, key, obs_seq, compute_up_to, theta, phi):

#         state_seq = self.q.compute_state_seq(obs_seq, compute_up_to, phi)

#         def _compute(key, obs_seq, state_seq):

#             def step(carry, x):

#                 key, obs, idx = x

#                 def false_fun(key, next_sample, obs, idx):
#                     return next_sample, 0.0            





class LinearGaussianELBO:

    def __init__(self, p:HMM, q:LinearGaussianHMM):
        self.p = p
        self.q = q
        
    def __call__(self, obs_seq, compute_up_to, theta:HMM.Params, phi:HMM.Params):

        def step(carry, x):
                
            state, kl_term = carry 
            idx, obs = x

            def false_fun(state, kl_term, obs):
                return (state, kl_term), None 
            
            def true_fun(state, kl_term, obs):

                q_backwd_params = self.q.backwd_params_from_state(state, phi)

                kl_term = expect_quadratic_term_under_backward(kl_term, q_backwd_params) \
                        + transition_term_integrated_under_backward(q_backwd_params, theta.transition) \
                        + get_tractable_emission_term(obs, theta.emission)


                kl_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_params.noise.scale.log_det) +  0.5 * self.p.state_dim
                new_state = self.q.new_state(obs, state, phi)
                return (new_state, kl_term), None

            return lax.cond(idx <= compute_up_to, true_fun, false_fun, state, kl_term, obs)


        kl_term = quadratic_term_from_log_gaussian(theta.prior) + get_tractable_emission_term(obs_seq[0], theta.emission)
        state = self.q.init_state(obs_seq[0], phi)
    
    
        (state, kl_term) = lax.scan(step, 
                                init=(state, kl_term), 
                                xs=(jnp.arange(1, len(obs_seq)), obs_seq[1:]))[0]

        q_last_filt_params = self.q.filt_params_from_state(state, phi)
        
        kl_term = expect_quadratic_term_under_gaussian(kl_term, q_last_filt_params) \
                    - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_params.scale.log_det) \
                    + 0.5*self.p.state_dim

        return kl_term / len(obs_seq)

def check_linear_gaussian_elbo(p:LinearGaussianHMM, num_seqs, seq_length):
    key_params, key_gen = jax.random.split(jax.random.PRNGKey(0), 2)
    theta = p.get_random_params(key_params)

    seqs = p.sample_multiple_sequences(key_gen, theta, num_seqs, seq_length)[1]

    elbo = LinearGaussianELBO(p,p)

    evidence_reference = vmap(lambda seq: p.likelihood_seq(seq, theta))(seqs) / len(seqs[0])
    theta = p.format_params(theta)
    evidence_elbo = vmap(lambda seq: elbo(seq, len(seq)-1, theta, theta))(seqs)

    print('ELBO sanity check:',jnp.mean(jnp.abs(evidence_elbo - evidence_reference)))

def check_general_elbo(mc_key, p:LinearGaussianHMM, num_seqs, seq_length, num_samples):

    key_params, key_gen = jax.random.split(jax.random.PRNGKey(0), 2)
    theta = p.get_random_params(key_params)

    seqs = p.sample_multiple_sequences(key_gen, theta, num_seqs, seq_length)[1]
    mc_keys = jax.random.split(mc_key, num_seqs)
    elbo = GeneralBackwardELBO(p,p,num_samples)

    evidence_reference = vmap(lambda seq: p.likelihood_seq(seq, theta))(seqs)
    
    theta = p.format_params(theta)
    evidence_elbo = vmap(lambda key, seq: elbo(key, seq, theta, theta))(mc_keys, seqs)
    print('ELBO sanity check:',jnp.mean(jnp.abs(evidence_elbo - evidence_reference)))


