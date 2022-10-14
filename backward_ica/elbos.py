import jax
from jax import vmap, lax, numpy as jnp
from .hmm import *
from .utils import *

def get_keys(key, num_seqs, num_epochs):
    keys = jax.random.split(key, num_seqs * num_epochs)
    keys = jnp.array(keys).reshape(num_epochs, num_seqs,-1)
    return keys

def get_dummy_keys(key, num_seqs, num_epochs): 
    return jnp.empty((num_epochs, num_seqs, 1))


class GeneralForwardELBO:

    def __init__(self, p:HMM, q:TwoFilterSmoother, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples 

    def __call__(self, key, obs_seq, theta, phi):

        def _monte_carlo_sample(key, obs_seq, init_law_params, forward_params_seq):
            keys = jax.random.split(key, obs_seq.shape[0])
            init_sample = self.q.marginal_dist.sample(keys[0], init_law_params)
            init_term = self.p.emission_kernel.logpdf(obs_seq[0], init_sample, theta.emission) \
                        + self.p.prior_dist.logpdf(init_sample, theta.prior) \
                        - self.q.marginal_dist.logpdf(init_sample, init_law_params)

            def _sample_step(prev_sample, x):

                key, obs, forward_params = x
                sample = self.q.forward_kernel.sample(key, prev_sample, forward_params)
                emission_term_p = self.p.emission_kernel.logpdf(obs, sample, theta.emission)
                transition_term_p = self.p.transition_kernel.logpdf(sample, prev_sample, theta.transition)
                fwd_term_q = -self.q.forward_kernel.logpdf(sample, prev_sample, forward_params)

                return sample, emission_term_p + transition_term_p + fwd_term_q

            init_sample, terms = lax.scan(f=_sample_step, 
                                        init=init_sample, 
                                        xs=(keys[1:], obs_seq[1:], forward_params_seq), reverse=False)


            return jnp.sum(terms) + init_term 


        parallel_sampler = vmap(_monte_carlo_sample, in_axes=(0,None,None,None))

        keys = jax.random.split(key, self.num_samples)

        state_seq = self.q.compute_state_seq(obs_seq, phi)
        backwd_variables_seq = self.q.compute_backwd_variables_seq(state_seq, phi)
        init_law_params = self.q.compute_marginal(self.q.init_filt_params(tree_get_idx(0, state_seq), phi), 
                                                tree_get_idx(0, backwd_variables_seq))
        forward_params_seq = vmap(self.q.forward_params_from_backwd_var, in_axes=(0,None))(tree_dropfirst(backwd_variables_seq), phi)


        mc_samples = parallel_sampler(keys, 
                                    obs_seq, 
                                    init_law_params,
                                    forward_params_seq)

        return jnp.mean(mc_samples)

class GeneralBackwardELBO:

    def __init__(self, p:HMM, q:BackwardSmoother, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples

    def __call__(self, key, obs_seq, theta:HMM.Params, phi):

        def _monte_carlo_sample(key, obs_seq, terminal_law_params, backwd_params_seq):

            keys = jax.random.split(key, obs_seq.shape[0])
            last_sample = self.q.filt_dist.sample(keys[-1], terminal_law_params)

            last_term = -self.q.filt_dist.logpdf(last_sample, terminal_law_params) \
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


        state_seq = self.q.compute_state_seq(obs_seq, phi)

        mc_samples = parallel_sampler(keys, 
                                    obs_seq, 
                                    self.q.filt_params_from_state(tree_get_idx(-1, state_seq), phi), 
                                    self.q.compute_backwd_params_seq(state_seq, phi))
        return jnp.mean(mc_samples)

class GeneralBackwardELBO:

    def __init__(self, p:HMM, q:BackwardSmoother, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples

    def __call__(self, key, obs_seq, theta:HMM.Params, phi):

        def _monte_carlo_sample(key, obs_seq, terminal_law_params, backwd_params_seq):

            keys = jax.random.split(key, obs_seq.shape[0])
            last_sample = self.q.filt_dist.sample(keys[-1], terminal_law_params)

            last_term = -self.q.filt_dist.logpdf(last_sample, terminal_law_params) \
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


        state_seq = self.q.compute_state_seq(obs_seq, phi)

        mc_samples = parallel_sampler(keys, 
                                    obs_seq, 
                                    self.q.filt_params_from_state(tree_get_idx(-1, state_seq), phi), 
                                    self.q.compute_backwd_params_seq(state_seq, phi))
        return jnp.mean(mc_samples)

class OnlineGeneralBackwardELBO:

    def __init__(self, p:HMM, q:BackwardSmoother, normalizer, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples
        self.normalizer = normalizer

    def __call__(self, key, obs_seq, theta:HMM.Params, phi):

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

    def __call__(self, key, obs_seq, theta:HMM.Params, phi):

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

    def __call__(self, key, obs_seq, theta:HMM.Params, phi):

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
            
    def __call__(self, key, obs_seq, theta:HMM.Params, phi):

        
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
            
    def __call__(self, key, obs_seq, theta:HMM.Params, phi):

        
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
        
    def __call__(self, obs_seq, theta:HMM.Params, phi:HMM.Params):

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


