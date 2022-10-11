import jax
import optax
from jax import tree_flatten, vmap, lax, numpy as jnp
from .hmm import *
from .utils import *
import tensorboard
import tensorflow as tf
import seaborn as sns
from backward_ica.smc import exp_and_normalize
config.update('jax_enable_x64',True)
import io

def get_keys(key, num_seqs, num_epochs):
    keys = jax.random.split(key, num_seqs * num_epochs)
    keys = jnp.array(keys).reshape(num_epochs, num_seqs,-1)
    return keys

def get_dummy_keys(key, num_seqs, num_epochs): 
    return jnp.empty((num_epochs, num_seqs, 1))


class GeneralBackwardELBO:

    def __init__(self, p:HMM, q:BackwardSmoother, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples

    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        filt_params_seq = self.q.compute_filt_params_seq(obs_seq, phi)
        backwd_params_seq = self.q.compute_backwd_params_seq(filt_params_seq, phi)

        def _monte_carlo_sample(key, obs_seq, last_filt_params:FiltState, backwd_params_seq):

            keys = jax.random.split(key, obs_seq.shape[0])
            last_sample = self.q.filt_dist.sample(keys[-1], last_filt_params.out)

            last_term = -self.q.filt_dist.logpdf(last_sample, last_filt_params.out) \
                    + self.p.emission_kernel.logpdf(obs_seq[-1], last_sample, theta.emission)

            def _sample_step(next_sample, x):
                
                key, obs, backwd_params = x

                sample = self.q.backwd_kernel.sample(key, next_sample, backwd_params)

                emission_term_p = self.p.emission_kernel.logpdf(obs, sample, theta.emission)

                transition_term_p = self.p.transition_kernel.logpdf(next_sample, sample, theta.transition)

                backwd_term_q = -self.q.backwd_kernel.logpdf(sample, next_sample, backwd_params)

                return sample, backwd_term_q + emission_term_p + transition_term_p
            
            init_sample, terms = lax.scan(_sample_step, init=last_sample, xs=(keys[:-1], obs_seq[:-1], backwd_params_seq), reverse=True)

            return self.p.prior_dist.logpdf(init_sample, theta.prior) + jnp.sum(terms) + last_term

        parallel_sampler = vmap(_monte_carlo_sample, in_axes=(0,None,None,None))

        keys = jax.random.split(key, self.num_samples)
        last_filt_params =  tree_get_idx(-1, filt_params_seq)
        mc_samples = parallel_sampler(keys, obs_seq, last_filt_params, backwd_params_seq)
        return jnp.mean(mc_samples)

class OnlineGeneralBackwardELBO:

    def __init__(self, p:HMM, q:BackwardSmoother, normalizer, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples
        self.normalizer = normalizer

    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        filt_params_seq = self.q.compute_filt_params_seq(obs_seq, phi)
        backwd_params_seq = self.q.compute_backwd_params_seq(filt_params_seq, phi)

        def sample_online(key, obs_seq, q_filt_params_seq, q_backwd_params_seq):

            def samples_and_log_probs(key, q_filt_params):
                samples = vmap(self.q.filt_dist.sample, in_axes=(0,None))(random.split(key, self.num_samples), q_filt_params.out)
                log_probs = vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples, q_filt_params.out)
                return samples, log_probs

            def additive_functional(obs, log_prob, sample, new_log_prob, new_sample, q_backwd_params):
                return self.p.transition_kernel.logpdf(new_sample, sample, theta.transition) \
                        + self.p.emission_kernel.logpdf(obs, new_sample, theta.emission) \
                        - self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_params) \
                        + log_prob \
                        - new_log_prob

            def init_functional(sample):
                return self.p.emission_kernel.logpdf(obs_seq[0], sample, theta.emission) \
                        + self.p.prior_dist.logpdf(sample, theta.prior)

            def update_tau(carry, x):

                tau, samples, log_probs = carry 
                key, obs, q_filt_params, q_backwd_params = x 
                new_samples, new_log_probs = samples_and_log_probs(key, q_filt_params)

                def update_component_tau(new_sample, new_log_prob):
                    def sum_component(sample, log_prob, tau_component):
                        log_weight = self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_params) - log_prob
                        component = tau_component + additive_functional(obs, 
                                                                    log_prob, 
                                                                    sample, 
                                                                    new_log_prob, 
                                                                    new_sample, 
                                                                    q_backwd_params)
                        return log_weight, component
                    log_weights, components = vmap(sum_component)(samples, log_probs, tau)

                    normalized_weights = self.normalizer(log_weights)
                    return normalized_weights, jnp.sum(normalized_weights * components)
                
                weights, new_tau = vmap(update_component_tau)(new_samples, new_log_probs) 

                return (new_tau, new_samples, new_log_probs), (new_samples, weights)

            key, subkey = random.split(key, 2)

            samples, log_probs = samples_and_log_probs(subkey, tree_get_idx(0, q_filt_params_seq))

            tau = vmap(init_functional)(samples)                            

            (tau, _ , _), (samples_seq, weights_seq) = lax.scan(update_tau, 
                                                                init=(tau, samples, log_probs), 
                                                                xs=(random.split(key, len(obs_seq)-1), 
                                                                    obs_seq[1:],
                                                                    tree_dropfirst(q_filt_params_seq),
                                                                    q_backwd_params_seq))

            return tau, tree_prepend(samples, samples_seq), weights_seq

        tau, samples_seq, weights_seq = sample_online(key, obs_seq, filt_params_seq, backwd_params_seq)

        return jnp.mean(tau), (samples_seq, weights_seq, filt_params_seq, backwd_params_seq)

class OnlineGeneralBackwardELBOV2:

    def __init__(self, p:HMM, q:BackwardSmoother, normalizer, num_samples=100):

        self.p = p
        self.q = q
        self.num_backwd_samples = 2
        self.num_samples = num_samples // self.num_backwd_samples

        self.normalizer = normalizer

    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        q_filt_params_seq = self.q.compute_filt_params_seq(obs_seq, phi)
        q_backwd_params_seq = self.q.compute_backwd_params_seq(q_filt_params_seq, phi)

        backwd_sampler = vmap(self.q.backwd_kernel.sample, in_axes=(0,None,None))

        def q_filt_log_probs(samples, q_filt_params):
            return vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples, q_filt_params.out)

        def q_filt_samples_and_log_probs(key, q_filt_params):
            samples = vmap(self.q.filt_dist.sample, in_axes=(0,None))(random.split(key, self.num_samples), q_filt_params.out)
            log_probs = vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples, q_filt_params.out)
            return samples, log_probs

        def additive_functional(q_t_x_t, x_t, q_tp1_x_tp1, x_tp1, y_tp1, q_t_tp1_params):
            return self.p.transition_kernel.logpdf(x_tp1, x_t, theta.transition) \
                    + self.p.emission_kernel.logpdf(y_tp1, x_tp1, theta.emission) \
                    + q_t_x_t \
                    - self.q.backwd_kernel.logpdf(x_t, x_tp1, q_t_tp1_params) \
                    - q_tp1_x_tp1

        def init_functional(x_0, q_0_x_0, x_1, q_1_x_1, y_0, y_1, q_0_1_params):

            return self.p.emission_kernel.logpdf(y_0, x_0, theta.emission) \
                    + self.p.emission_kernel.logpdf(y_1, x_1, theta.emission) \
                    + self.p.transition_kernel.logpdf(x_1, x_0, theta.transition) \
                    + self.p.prior_dist.logpdf(x_0, theta.prior) \
                    + q_0_x_0 \
                    - self.q.backwd_kernel.logpdf(x_1, x_0, q_0_1_params) \
                    - q_1_x_1
                
        def init(key, y_0, y_1, q_0_params, q_1_params, q_0_1_params):

            key, subkey = random.split(key, 2)

            xi_0, q_0_xi_0 = q_filt_samples_and_log_probs(subkey, q_0_params)

            def init_tau(x_0):
                return self.p.prior_dist.logpdf(x_0, theta.prior) + self.p.emission_kernel.logpdf(y_0, x_0, theta.emission)

            tau_0 = vmap(init_tau)(xi_0)

            key, subkey = random.split(key, 2)

            xi_1, q_1_xi_1 = q_filt_samples_and_log_probs(subkey, q_1_params)

            def T_1(key, x_1, q_1_x_1): 
                xi_0_1 = backwd_sampler(random.split(key, self.num_backwd_samples), x_1, q_0_1_params)
                q_0_xi_0_1 = q_filt_log_probs(xi_0_1, q_0_params)
                return jnp.mean(vmap(fun=init_functional, in_axes=(0, 0, None, None, None, None, None))(xi_0_1, 
                                                                                                    q_0_xi_0_1,
                                                                                                    x_1, 
                                                                                                    q_1_x_1,
                                                                                                    y_0, 
                                                                                                    y_1, 
                                                                                                    q_0_1_params))
            tau_1 = vmap(T_1)(random.split(key, self.num_samples), xi_1, q_1_xi_1)


            return tau_0, xi_0, q_0_xi_0, q_0_1_params, tau_1, q_1_params, xi_1, q_1_xi_1, y_1


        def step(carry, x):

            tau_tm1, xi_tm1, q_tm1_xi_tm1, q_tm1_t_params, tau_t, q_t_params, xi_t, q_t_xi_t, y_t = carry
            key, y_tp1, q_tp1_params, q_t_tp1_params = x

            key, subkey = random.split(key, 2)

            def tilde_T_t(x_t, q_t_x_t):
                def tilde_T_t_sum_component(xi_tm1_j, q_tm1_xi_tm1_j, tau_tm1_j):
                    importance_log_weight = self.q.backwd_kernel.logpdf(xi_tm1_j, x_t, q_t_tp1_params) - q_tm1_xi_tm1_j
                    tau_component_t_tp1 = tau_tm1_j + additive_functional(q_tm1_xi_tm1_j, 
                                                                        xi_tm1_j, 
                                                                        q_t_x_t, 
                                                                        x_t, 
                                                                        y_t, 
                                                                        q_tm1_t_params)

                    return importance_log_weight, tau_component_t_tp1
                log_weights, sum_components = vmap(tilde_T_t_sum_component)(xi_tm1, q_tm1_xi_tm1, tau_tm1)

                return jnp.sum(self.normalizer(log_weights) * sum_components)

            xi_tp1, q_tp1_xi_tp1 = q_filt_samples_and_log_probs(subkey, q_tp1_params)

            def T_tp1(key, x_tp1, q_tp1_x_tp1):

                xi_t_tp1 = backwd_sampler(random.split(key, self.num_backwd_samples), x_tp1, q_t_tp1_params)
                q_t_xi_t_tp1 = q_filt_log_probs(xi_t_tp1, q_t_params)

                def T_tp1_sum_component(xi_t_tp1_j, q_t_xi_t_tp1_j):
                    return tilde_T_t(xi_t_tp1_j, q_t_xi_t_tp1_j) + additive_functional(q_t_xi_t_tp1_j, 
                                                                                        xi_t_tp1_j,
                                                                                        q_tp1_x_tp1,
                                                                                        x_tp1,
                                                                                        y_tp1,
                                                                                        q_t_tp1_params)

                return jnp.mean(vmap(T_tp1_sum_component)(xi_t_tp1, q_t_xi_t_tp1))

            tau_tp1 = vmap(T_tp1)(random.split(key, self.num_samples), xi_tp1, q_tp1_xi_tp1)
            
            
            return (tau_t, xi_t, q_t_xi_t, q_t_tp1_params, tau_tp1, q_tp1_params, xi_tp1, q_tp1_xi_tp1, y_tp1),  None

        key, subkey = random.split(key, 2)
        
        tau_T = lax.scan(f=step, 
                        init=init(subkey, 
                                obs_seq[0], 
                                obs_seq[1], 
                                tree_get_idx(0, q_filt_params_seq), 
                                tree_get_idx(1, q_filt_params_seq),
                                tree_get_idx(0, q_backwd_params_seq)),
                        xs=(random.split(key, len(obs_seq)-2), 
                            obs_seq[2:],
                            tree_get_slice(2, None, q_filt_params_seq),
                            tree_get_slice(1, None, q_backwd_params_seq)))[0][0]


        return jnp.mean(tau_T), (0,0,0,0)

class OnlineGeneralBackwardELBOV3:

    def __init__(self, p:HMM, q:BackwardSmoother, normalizer, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples
        self.normalizer = normalizer

    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        filt_params_seq = self.q.compute_filt_params_seq(obs_seq, phi)
        backwd_params_seq = self.q.compute_backwd_params_seq(filt_params_seq, phi)

        def sample_online(key, obs_seq, q_filt_params_seq, q_backwd_params_seq):

            def samples_and_log_probs(key, q_filt_params):
                samples = vmap(self.q.filt_dist.sample, in_axes=(0,None))(random.split(key, self.num_samples), q_filt_params.out)
                log_probs = vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples, q_filt_params.out)
                return samples, log_probs

            def additive_functional(obs, log_prob, sample, new_log_prob, new_sample, q_backwd_params):
                return self.p.transition_kernel.logpdf(new_sample, sample, theta.transition) \
                        + self.p.emission_kernel.logpdf(obs, new_sample, theta.emission) \
                        - self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_params) \
                        + log_prob \
                        - new_log_prob

            def init_functional(sample):
                return self.p.emission_kernel.logpdf(obs_seq[0], sample, theta.emission) \
                        + self.p.prior_dist.logpdf(sample, theta.prior)

            def update_tau(carry, x):

                tau, samples, log_probs = carry 
                key, obs, q_filt_params, q_backwd_params = x 
                new_samples, new_log_probs = samples_and_log_probs(key, q_filt_params)

                def update_component_tau(new_sample, new_log_prob):
                    def sum_component(sample, log_prob, tau_component):
                        log_weight = self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_params) - log_prob
                        component = tau_component + additive_functional(obs, 
                                                                    log_prob, 
                                                                    sample, 
                                                                    new_log_prob, 
                                                                    new_sample, 
                                                                    q_backwd_params)
                        return log_weight, component
                    log_weights, components = vmap(sum_component)(samples, log_probs, tau)

                    normalized_weights = self.normalizer(log_weights)
                    return normalized_weights, jnp.sum(normalized_weights * components)
                
                weights, new_tau = vmap(update_component_tau)(new_samples, new_log_probs) 

                return (new_tau, new_samples, new_log_probs), (new_samples, weights)

            key, subkey = random.split(key, 2)

            samples, log_probs = samples_and_log_probs(subkey, tree_get_idx(0, q_filt_params_seq))

            tau = vmap(init_functional)(samples)                            

            (tau, _ , _), (samples_seq, weights_seq) = lax.scan(update_tau, 
                                                                init=(tau, samples, log_probs), 
                                                                xs=(random.split(key, len(obs_seq)-1), 
                                                                    obs_seq[1:],
                                                                    tree_dropfirst(q_filt_params_seq),
                                                                    q_backwd_params_seq))

            return tau, tree_prepend(samples, samples_seq), weights_seq

        tau, samples_seq, weights_seq = sample_online(key, obs_seq, filt_params_seq, backwd_params_seq)

        return jnp.mean(tau), (samples_seq, weights_seq, filt_params_seq, backwd_params_seq)

class OnlineBackwardLinearELBO:

    def __init__(self, p:HMM, q:LinearBackwardSmoother, normalizer, num_samples=200):
        
        self.p = p
        self.q = q
        self.num_samples = num_samples
        self.normalizer = normalizer
            
    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        
        kl_term = quadratic_term_from_log_gaussian(theta.prior) #+ get_tractable_emission_term(obs_seq[0], theta.emission)

        q_filt_params = self.q.init_filt_params(obs_seq[0], phi)

        def V_step(carry, x):

            q_filt_params, kl_term = carry
            obs = x

            q_backwd_params = self.q.new_backwd_params(q_filt_params, phi)

            kl_term = expect_quadratic_term_under_backward(kl_term, q_backwd_params) \
                    + transition_term_integrated_under_backward(q_backwd_params, theta.transition)

            kl_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_params.noise.scale.log_det) +  0.5 * self.p.state_dim
            q_filt_params = self.q.new_filt_params(obs, q_filt_params, phi)


            return (q_filt_params, kl_term), (q_filt_params, q_backwd_params)
    
        (q_last_filt_params, kl_term), (q_filt_params_seq, q_backwd_params_seq) = lax.scan(V_step, 
                                                init=(q_filt_params, kl_term), 
                                                xs=obs_seq[1:])

        q_filt_params_seq = tree_prepend(q_filt_params, q_filt_params_seq)


        kl_term = expect_quadratic_term_under_gaussian(kl_term, q_last_filt_params.out) \
                - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_params.out.scale.log_det) \
                + 0.5*self.p.state_dim

        def sample_online(key, obs_seq, q_filt_params_seq, q_backwd_params_seq):

            def samples_and_log_probs(key, q_filt_params):
                samples = vmap(self.q.filt_dist.sample, in_axes=(0,None))(random.split(key, self.num_samples), q_filt_params.out)
                log_probs = vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples, q_filt_params.out)
                return samples, log_probs

            key, subkey = random.split(key, 2)

            samples, log_probs = samples_and_log_probs(subkey, tree_get_idx(0, q_filt_params_seq))
            tau = vmap(self.p.emission_kernel.logpdf, in_axes=(None, 0, None))(obs_seq[0], samples, theta.emission)
            # tau = jnp.zeros(self.num_samples)
            def update_tau(carry, x):

                tau, samples, log_probs = carry 
                key, obs, q_filt_params, q_backwd_params = x 
                new_samples, new_log_probs = samples_and_log_probs(key, q_filt_params)

                def update_component_tau(new_sample):
                    def sum_component(sample, log_prob, tau_component):
                        log_weight = self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_params) - log_prob
                        component = tau_component + self.p.emission_kernel.logpdf(obs, new_sample, theta.emission)
                        return log_weight, component
                    log_weights, components = vmap(sum_component)(samples, log_probs, tau)

                    return jnp.sum(self.normalizer(log_weights) * components), log_weights
                
                new_tau, new_log_weights = vmap(update_component_tau)(new_samples) 

                return (new_tau, new_samples, new_log_probs), new_log_weights
            
            return lax.scan(update_tau, 
                            init=(tau, samples, log_probs), 
                            xs=(random.split(key, len(obs_seq)-1), 
                                obs_seq[1:],
                                tree_dropfirst(q_filt_params_seq),
                                q_backwd_params_seq))

        (tau, _, _), log_weights = sample_online(key, obs_seq, q_filt_params_seq, q_backwd_params_seq)

        return kl_term + jnp.mean(tau), log_weights

class BackwardLinearELBO:

    def __init__(self, p:HMM, q:LinearBackwardSmoother, num_samples=200):
        
        self.p = p
        self.q = q
        self.num_samples = num_samples
            
    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        
        kl_term = quadratic_term_from_log_gaussian(theta.prior) #+ get_tractable_emission_term(obs_seq[0], theta.emission)

        q_filt_params = self.q.init_filt_params(obs_seq[0], phi)

        def V_step(carry, x):

            q_filt_params, kl_term = carry
            obs = x

            q_backwd_params = self.q.new_backwd_params(q_filt_params, phi)

            kl_term = expect_quadratic_term_under_backward(kl_term, q_backwd_params) \
                    + transition_term_integrated_under_backward(q_backwd_params, theta.transition)

            kl_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_params.noise.scale.log_det) +  0.5 * self.p.state_dim
            q_filt_params = self.q.new_filt_params(obs, q_filt_params, phi)


            return (q_filt_params, kl_term), q_backwd_params
    
        (q_last_filt_params, kl_term), q_backwd_params_seq = lax.scan(V_step, 
                                                init=(q_filt_params, kl_term), 
                                                xs=obs_seq[1:])


        kl_term =  expect_quadratic_term_under_gaussian(kl_term, q_last_filt_params.out) \
                    - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_params.out.scale.log_det) \
                    + 0.5*self.p.state_dim


        marginals = self.q.compute_marginals(q_last_filt_params, q_backwd_params_seq)

        def sample_one_path(key, obs_seq, marginal_params_seq):

            def term_at_t(key, obs, marginal_params):
                sample = Gaussian.sample(key, marginal_params)
                return self.p.emission_kernel.logpdf(obs, sample, theta.emission)

            keys = jax.random.split(key, len(obs_seq))

            return jnp.sum(vmap(term_at_t)(keys, obs_seq, marginal_params_seq))


        parallel_sampler = vmap(sample_one_path, in_axes=(0,None,None))
        
        keys = jax.random.split(key, self.num_samples)
        
        mc_samples = parallel_sampler(keys, obs_seq, marginals)

        return kl_term + jnp.mean(mc_samples)

class LinearGaussianELBO:

    def __init__(self, p:HMM, q:LinearGaussianHMM):
        self.p = p
        self.q = q
        
    def __call__(self, obs_seq, theta:HMMParams, phi:HMMParams):

        result = quadratic_term_from_log_gaussian(theta.prior) + get_tractable_emission_term(obs_seq[0], theta.emission)


        q_filt_params = self.q.init_filt_params(obs_seq[0], phi)

        def V_step(state, obs):

            q_filt_params, kl_term = state
            q_backwd_params = self.q.new_backwd_params(q_filt_params, phi)

            kl_term = expect_quadratic_term_under_backward(kl_term, q_backwd_params) \
                    + transition_term_integrated_under_backward(q_backwd_params, theta.transition) \
                    + get_tractable_emission_term(obs, theta.emission)


            kl_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_params.noise.scale.log_det) +  0.5 * self.p.state_dim
            q_filt_params = self.q.new_filt_params(obs, q_filt_params, phi)

            return (q_filt_params, kl_term), q_backwd_params
    
        (q_last_filt_params, result) = lax.scan(V_step, 
                                                init=(q_filt_params, result), 
                                                xs=obs_seq[1:])[0]


        return expect_quadratic_term_under_gaussian(result, q_last_filt_params.out) \
                    - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_params.out.scale.log_det) \
                    + 0.5*self.p.state_dim



def winsorize_grads():
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        flattened_updates = jnp.concatenate([arr.flatten() for arr in tree_flatten(updates)[0]])
        high_value = jnp.sort(jnp.abs(flattened_updates))[int(0.90*flattened_updates.shape[0])]
        return jax.tree_map(lambda x: jnp.clip(x, -high_value, high_value), updates), ()
    return optax.GradientTransformation(init_fn, update_fn)

class SVITrainer:

    def __init__(self, p:HMM, 
                theta_star,
                q:BackwardSmoother, 
                optimizer, 
                learning_rate, 
                num_epochs, 
                batch_size, 
                num_samples=1, 
                force_full_mc=False,
                frozen_params=None,
                online=False,
                sweep_sequence=False):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.q = q 
        # self.q.print_num_params()
        self.p = p 
        
        self.theta_star = theta_star
        self.frozen_params = frozen_params

        self.trainable_params = tree_map(lambda x: x == '', self.frozen_params)
        self.fixed_params = tree_map(lambda x: x != '', self.frozen_params)

        base_optimizer = optax.apply_if_finite(optax.masked(getattr(optax, optimizer)(learning_rate), 
                                                            self.trainable_params), 
                                            max_consecutive_errors=10)

        zero_grads_optimizer = optax.masked(optax.set_to_zero(), self.fixed_params)

        self.optimizer = optax.chain(zero_grads_optimizer, base_optimizer)
        self.sweep_sequence = sweep_sequence
        # format_params = lambda params: self.q.format_params(params)


        if online:
            self.elbo = OnlineGeneralBackwardELBO(self.p, self.q, exp_and_normalize, num_samples)
            self.get_montecarlo_keys = get_keys
            self.loss = lambda key, data, params: -self.elbo(key, data, self.p.format_params(self.theta_star), q.format_params(params))[0]
        else:
            if force_full_mc: 
                self.elbo = GeneralBackwardELBO(self.p, self.q, num_samples)
                self.get_montecarlo_keys = get_keys
                self.loss = lambda key, data, params: -self.elbo(key, data, self.p.format_params(self.theta_star), q.format_params(params))
            else:
                if isinstance(self.p, LinearGaussianHMM):
                    self.elbo = LinearGaussianELBO(self.p, self.q)
                    self.get_montecarlo_keys = get_dummy_keys
                    self.loss = lambda key, data, params: -self.elbo(data, self.p.format_params(self.theta_star), q.format_params(params))
                elif isinstance(self.q, LinearBackwardSmoother) and self.p.transition_kernel.map_type == 'linear':
                    self.elbo = BackwardLinearELBO(self.p, self.q, num_samples)
                    self.get_montecarlo_keys = get_keys
                    self.loss = lambda key, data, params: -self.elbo(key, data, self.p.format_params(self.theta_star), q.format_params(params))
                else:
                    self.elbo = GeneralBackwardELBO(self.p, self.q, num_samples)
                    self.get_montecarlo_keys = get_keys
                    self.loss = lambda key, data, params: -self.elbo(key, data, self.p.format_params(self.theta_star), q.format_params(params))

    def fit(self, key_params, key_batcher, key_montecarlo, data, log_writer=None, args=None):


        num_seqs = data.shape[0]
        seq_length = len(data[0])

        # theta = self.p.get_random_params(key_theta, args)
        params = self.q.get_random_params(key_params, args)

        params = tree_map(lambda param, frozen_param: param if frozen_param == '' else frozen_param, 
                        params, 
                        self.frozen_params)

        opt_state = self.optimizer.init(params)
        subkeys = self.get_montecarlo_keys(key_montecarlo, num_seqs, self.num_epochs)

        if self.sweep_sequence: 
            # timesteps = jnp.arange(0, seq_length)
            def step(carry, x):

                def batch_step(params, opt_state, batch, keys):
                    avg_elbo_batch_timesteps = jnp.empty((seq_length-1,))
                    for i, timestep in enumerate(range(2,seq_length+1)):
                        batch_up_to_timestep = jax.lax.dynamic_slice_in_dim(batch, 0, timestep, axis=1)
                        neg_elbo_values, grads = jax.vmap(jax.value_and_grad(self.loss, argnums=2), in_axes=(0,0,None))(keys, batch_up_to_timestep, params)
                        avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), grads)
                        updates, opt_state = self.optimizer.update(avg_grads, opt_state, params)
                        params = optax.apply_updates(params, updates)
                        avg_elbo_batch_timesteps = avg_elbo_batch_timesteps.at[i].set(-jnp.mean(neg_elbo_values / batch_up_to_timestep.shape[1]))

                    return params, opt_state, jnp.mean(avg_elbo_batch_timesteps)

                data, params, opt_state, subkeys_epoch = carry
                batch_start = x
                batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
                batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)

                params, opt_state, avg_elbo_batch = batch_step(params, opt_state, batch_obs_seq, batch_keys)

                return (data, params, opt_state, subkeys_epoch), avg_elbo_batch
        else: 
            def step(carry, x):
                def batch_step(params, opt_state, batch, keys):

                    neg_elbo_values, grads = jax.vmap(jax.value_and_grad(self.loss, argnums=2), in_axes=(0,0,None))(keys, batch, params)
                    avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), grads)
                    
                    updates, opt_state = self.optimizer.update(avg_grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return params, \
                        opt_state, \
                        -jnp.mean(neg_elbo_values / seq_length)

                data, params, opt_state, subkeys_epoch = carry
                batch_start = x
                batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
                batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)
                params, opt_state, avg_elbo_batch = batch_step(params, opt_state, batch_obs_seq, batch_keys)
                return (data, params, opt_state, subkeys_epoch), avg_elbo_batch


        avg_elbos = []
        all_params = []
        batch_start_indices = jnp.arange(0, num_seqs, self.batch_size)

        t = tqdm(total=self.num_epochs, desc='Epoch')
        for epoch_nb in range(self.num_epochs):
            t.update(1)
            subkeys_epoch = subkeys[epoch_nb]
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            
            data = jax.random.permutation(subkey_batcher, data)
        

            (_ , params, opt_state, _), avg_elbo_batches = jax.lax.scan(f=step,  
                                                                        init=(data, params, opt_state, subkeys_epoch), 
                                                                        xs = batch_start_indices)


            avg_elbo_epoch = jnp.mean(avg_elbo_batches)
            t.set_postfix({'Avg ELBO epoch':avg_elbo_epoch})

            # avg_grads_batches = [grad for mask, grad in zip(tree_flatten(self.trainable_params)[0], 
            #                                                 tree_flatten(avg_grads_batches)[0]) 
            #                                             if mask]



            
            if log_writer is not None:
                with log_writer.as_default():
                    tf.summary.scalar('Epoch ELBO', avg_elbo_epoch, epoch_nb)
                    for batch_nb, avg_elbo_batch in enumerate(avg_elbo_batches):
                        tf.summary.scalar('Minibatch ELBO', avg_elbo_batch, epoch_nb*len(batch_start_indices) + batch_nb)
                        # avg_grads_batch = jnp.concatenate([grad[batch_nb].flatten() for grad in avg_grads_batches])
                        # sns.violinplot(avg_grads_batch)
                        # sns.swarmplot(avg_grads_batch)
                        # plt.savefig(os.path.join('grads',f'{epoch_nb*len(batch_start_indices) + batch_nb}'))
                        # tf.summary.image('Minibatch grads histogram', plot_to_image(plt.gcf()), epoch_nb*len(batch_start_indices) + batch_nb)
                        # plt.clf()
                        # tf.summary.histogram('Minibatch grads', avg_grads_batch, epoch_nb*len(batch_start_indices) + batch_nb)
            avg_elbos.append(avg_elbo_epoch)
            all_params.append(params)
        t.close()
                    
        return all_params, avg_elbos

    def multi_fit(self, key_params, key_batcher, key_montecarlo, data, num_fits, store_every=None, log_dir='', args=None):


        all_avg_elbos = []
        all_params = []
        best_elbos = []
        best_epochs = []
        
        print('Starting training...')
        
        tensorboard_subdir = os.path.join(log_dir, 'tensorboard_logs')
        os.makedirs(tensorboard_subdir, exist_ok=True)
        for fit_nb, subkey_params in enumerate(jax.random.split(key_params, num_fits)):
            log_writer = tf.summary.create_file_writer(os.path.join(tensorboard_subdir, f'fit_{fit_nb}'))

            print(f'Fit {fit_nb+1}/{num_fits}')
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            key_montecarlo, subkey_montecarlo = jax.random.split(key_montecarlo, 2)

            params, avg_elbos = self.fit(subkey_params, subkey_batcher, subkey_montecarlo, data, log_writer, args)

            best_epoch = jnp.nanargmax(jnp.array(avg_elbos))
            best_epochs.append(best_epoch)
            best_elbo = avg_elbos[best_epoch]
            best_elbos.append(best_elbo)
            print(f'Best ELBO {best_elbo:.3f} at epoch {best_epoch}')
        
            if store_every is not None:
                selected_epochs = list(range(0, self.num_epochs, store_every))
                all_params.append({epoch_nb:params[epoch_nb] for epoch_nb in selected_epochs})

            else: 
                all_params.append(params[best_epoch])
            all_avg_elbos.append(avg_elbos)


        best_optim = jnp.argmax(jnp.array(best_elbos))
        print(f'Best fit is {best_optim+1}.')
        best_params = all_params[best_optim]

        if store_every is not None: 
            return best_params, all_avg_elbos[best_optim]
        else: 
            return best_params, (best_optim, best_epochs, all_avg_elbos)

    def profile(self, key, data, theta):

        params_key, monte_carlo_key = jax.random.split(key, 2)
        phi = self.q.get_random_params(params_key)

        if self.use_johnson: 
            aux_params = self.aux_init_params(params_key, data[0][0])
        else:
            aux_params = None        
        if self.use_johnson: 
            loss = lambda seq, key, phi, aux_params: self.loss(seq, key, self.p.format_params(theta), self.q.format_params(phi), aux_params)
            params = (phi, aux_params)
        else: 
            loss = lambda seq, key, phi: self.loss(seq, key, self.p.format_params(theta), self.q.format_params(phi), None)
            params = phi

        num_seqs = data.shape[0]
        subkeys = self.get_montecarlo_keys(monte_carlo_key, num_seqs, 1).squeeze()

        @jit
        def step(params, batch, keys):
            return jax.vmap(loss, in_axes=(0,0,None))(batch, keys, params)

        step(params, data[:self.batch_size], subkeys[:self.batch_size])

        with jax.profiler.trace('./profiling/'):
            print(step(params, data[:2], subkeys[:2]))

    def check_elbo(self, data, theta):
        if isinstance(self.p, LinearGaussianHMM):
            print('Checking ELBO quality...')

            avg_evidences = vmap(jit(lambda seq: self.p.likelihood_seq(seq, theta)))(data)
            theta = self.p.format_params(theta)
            if isinstance(self.elbo, LinearGaussianELBO):
                elbo = jit(lambda key, seq:self.elbo(seq, theta, theta))
            else: 
                elbo = jit(lambda key, seq: self.elbo(key, seq, theta, theta))
            keys = jax.random.split(jax.random.PRNGKey(0), data.shape[0])
            avg_elbos = vmap(elbo)(keys, data)
            print('Avg error with Kalman evidence:', jnp.mean(jnp.abs(avg_evidences-avg_elbos)))


def check_linear_gaussian_elbo(p:LinearGaussianHMM, num_seqs, seq_length):
    key_params, key_gen = jax.random.split(jax.random.PRNGKey(0), 2)
    theta = p.get_random_params(key_params)

    seqs = p.sample_multiple_sequences(key_gen, theta, num_seqs, seq_length)[1]

    elbo = LinearGaussianELBO(p,p)

    evidence_reference = vmap(lambda seq: p.likelihood_seq(seq, theta))(seqs)
    theta = p.format_params(theta)
    evidence_elbo = vmap(lambda seq: elbo(seq, theta, theta))(seqs)

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

def check_backward_linear_elbo(mc_key, p:LinearGaussianHMM, num_seqs, seq_length, num_samples):

    key_params, key_gen = jax.random.split(jax.random.PRNGKey(0), 2)
    theta = p.get_random_params(key_params)

    seqs = p.sample_multiple_sequences(key_gen, theta, num_seqs, seq_length)[1]
    mc_keys = jax.random.split(mc_key, num_seqs)
    elbo = BackwardLinearELBO(p,p,num_samples)

    evidence_reference = vmap(lambda seq: p.likelihood_seq(seq, theta))(seqs)
    
    theta = p.format_params(theta)
    evidence_elbo = vmap(lambda key, seq: elbo(key, seq, theta, theta))(mc_keys, seqs)
    print('ELBO sanity check:',jnp.mean(jnp.abs(evidence_elbo - evidence_reference)))
    # time0 = time() 
    # print('Grad:', grad_elbo(mc_keys, seqs, theta, theta))
    # print('Timing:', time() - time0)

    # evidence_elbo = vmap(lambda key, seq: elbo(key, seq, theta, theta))(mc_keys, seqs)
    # # print('ELBO:', evidence_elbo)
    # print('ELBO sanity check:',jnp.mean(jnp.abs(evidence_elbo - evidence_reference)))


