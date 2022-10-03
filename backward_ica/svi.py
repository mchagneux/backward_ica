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

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def constant_terms_from_log_gaussian(dim:int, log_det:float)->float:
    """Utility function to compute the log of the term that is against the exponential for a multivariate Normal

    Args:
        dim (int): the dimension of the support of the multivariate Normal
        det (float): the precomputed determinant of the covariance matrix 

    Returns:
        float: the value of the requested factor  
    """

    return -0.5*(dim * jnp.log(2*jnp.pi) + log_det)

def transition_term_integrated_under_backward(q_backwd_state, transition_params):
    # expectation of the quadratic form that appears in the log of the state transition density

    A = transition_params.map.w @ q_backwd_state.map.w - jnp.eye(transition_params.noise.scale.cov.shape[0])
    b = transition_params.map.w @ q_backwd_state.map.b + transition_params.map.b
    Omega = transition_params.noise.scale.prec
    
    result = -0.5 * QuadTerm.from_A_b_Omega(A, b, Omega)
    result.c += -0.5 * jnp.trace(transition_params.noise.scale.prec @ transition_params.map.w @ q_backwd_state.noise.scale.cov @ transition_params.map.w.T) \
                + constant_terms_from_log_gaussian(transition_params.noise.scale.cov.shape[0], transition_params.noise.scale.log_det)
    return result 

def expect_quadratic_term_under_backward(quad_form:QuadTerm, backwd_state):
    # the result is still a quadratic forms with new parameters, following the formula for expected values of quadratic forms  

    W = backwd_state.map.w.T @ quad_form.W @ backwd_state.map.w
    v = backwd_state.map.w.T @ (quad_form.v + (quad_form.W + quad_form.W.T) @ backwd_state.map.b)
    c = quad_form.c + jnp.trace(quad_form.W @ backwd_state.noise.scale.cov) + backwd_state.map.b.T @ quad_form.W @ backwd_state.map.b + quad_form.v.T @ backwd_state.map.b 

    return QuadTerm(W=W, v=v, c=c)

def expect_quadratic_term_under_gaussian(quad_form:QuadTerm, gaussian_params):
    return jnp.trace(quad_form.W @ gaussian_params.scale.cov) + quad_form.evaluate(gaussian_params.mean)

def quadratic_term_from_log_gaussian(gaussian_params):

    result = - 0.5 * QuadTerm(W=gaussian_params.scale.prec, 
                    v=-(gaussian_params.scale.prec + gaussian_params.scale.prec.T) @ gaussian_params.mean, 
                    c=gaussian_params.mean.T @ gaussian_params.scale.prec @ gaussian_params.mean)

    result.c += constant_terms_from_log_gaussian(gaussian_params.mean.shape[0], gaussian_params.scale.log_det)

    return result

def get_tractable_emission_term(obs, emission_params):
    A = emission_params.map.w
    b = emission_params.map.b - obs
    Omega = emission_params.noise.scale.prec
    emission_term = -0.5*QuadTerm.from_A_b_Omega(A, b, Omega)
    emission_term.c += constant_terms_from_log_gaussian(emission_params.noise.scale.cov.shape[0], emission_params.noise.scale.log_det)
    return emission_term

def get_tractable_emission_term_from_natparams(emission_natparams):
    eta1, eta2 = emission_natparams
    const = -0.25 * eta1.T @ jnp.linalg.solve(eta2, eta1) - 0.5 * jnp.log(jnp.linalg.det(-2*eta2)) - eta1.shape[0] * jnp.log(jnp.pi)
    return QuadTerm(W=eta2, 
                    v=eta1, 
                    c=const)


class GeneralBackwardELBO:

    def __init__(self, p:HMM, q:BackwardSmoother, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples

    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        filt_state_seq = self.q.compute_filt_state_seq(obs_seq, phi)
        backwd_state_seq = self.q.compute_kernel_state_seq(filt_state_seq, phi)

        def _monte_carlo_sample(key, obs_seq, last_filt_state:FiltState, backwd_state_seq):

            keys = jax.random.split(key, obs_seq.shape[0])
            last_sample = self.q.filt_dist.sample(keys[-1], last_filt_state.out)

            last_term = -self.q.filt_dist.logpdf(last_sample, last_filt_state.out) \
                    + self.p.emission_kernel.logpdf(obs_seq[-1], last_sample, theta.emission)

            def _sample_step(next_sample, x):
                
                key, obs, backwd_state = x

                sample = self.q.backwd_kernel.sample(key, next_sample, backwd_state)

                emission_term_p = self.p.emission_kernel.logpdf(obs, sample, theta.emission)

                transition_term_p = self.p.transition_kernel.logpdf(next_sample, sample, theta.transition)

                backwd_term_q = -self.q.backwd_kernel.logpdf(sample, next_sample, backwd_state)

                return sample, backwd_term_q + emission_term_p + transition_term_p
            
            init_sample, terms = lax.scan(_sample_step, init=last_sample, xs=(keys[:-1], obs_seq[:-1], backwd_state_seq), reverse=True)

            return self.p.prior_dist.logpdf(init_sample, theta.prior) + jnp.sum(terms) + last_term

        parallel_sampler = vmap(_monte_carlo_sample, in_axes=(0,None,None,None))

        keys = jax.random.split(key, self.num_samples)
        last_filt_state =  tree_get_idx(-1, filt_state_seq)
        mc_samples = parallel_sampler(keys, obs_seq, last_filt_state, backwd_state_seq)
        return jnp.mean(mc_samples)

class OnlineGeneralBackwardELBO:

    def __init__(self, p:HMM, q:BackwardSmoother, normalizer, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples
        self.normalizer = normalizer

    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        filt_state_seq = self.q.compute_filt_state_seq(obs_seq, phi)
        backwd_state_seq = self.q.compute_kernel_state_seq(filt_state_seq, phi)

        def sample_online(key, obs_seq, q_filt_state_seq, q_backwd_state_seq):

            def samples_and_log_probs(key, q_filt_state):
                samples = vmap(self.q.filt_dist.sample, in_axes=(0,None))(random.split(key, self.num_samples), q_filt_state.out)
                log_probs = vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples, q_filt_state.out)
                return samples, log_probs

            def additive_functional(obs, log_prob, sample, new_log_prob, new_sample, q_backwd_state):
                return self.p.transition_kernel.logpdf(new_sample, sample, theta.transition) \
                        + self.p.emission_kernel.logpdf(obs, new_sample, theta.emission) \
                        - self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_state) \
                        + log_prob \
                        - new_log_prob

            def init_functional(sample):
                return self.p.emission_kernel.logpdf(obs_seq[0], sample, theta.emission) \
                        + self.p.prior_dist.logpdf(sample, theta.prior)

            def update_tau(carry, x):

                tau, samples, log_probs = carry 
                key, obs, q_filt_state, q_backwd_state = x 
                new_samples, new_log_probs = samples_and_log_probs(key, q_filt_state)

                def update_component_tau(new_sample, new_log_prob):
                    def sum_component(sample, log_prob, tau_component):
                        log_weight = self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_state) - log_prob
                        component = tau_component + additive_functional(obs, 
                                                                    log_prob, 
                                                                    sample, 
                                                                    new_log_prob, 
                                                                    new_sample, 
                                                                    q_backwd_state)
                        return log_weight, component
                    log_weights, components = vmap(sum_component)(samples, log_probs, tau)

                    normalized_weights = self.normalizer(log_weights)
                    return normalized_weights, jnp.sum(normalized_weights * components)
                
                weights, new_tau = vmap(update_component_tau)(new_samples, new_log_probs) 

                return (new_tau, new_samples, new_log_probs), (new_samples, weights)

            key, subkey = random.split(key, 2)

            samples, log_probs = samples_and_log_probs(subkey, tree_get_idx(0, q_filt_state_seq))

            tau = vmap(init_functional)(samples)                            

            (tau, _ , _), (samples_seq, weights_seq) = lax.scan(update_tau, 
                                                                init=(tau, samples, log_probs), 
                                                                xs=(random.split(key, len(obs_seq)-1), 
                                                                    obs_seq[1:],
                                                                    tree_dropfirst(q_filt_state_seq),
                                                                    q_backwd_state_seq))

            return tau, tree_prepend(samples, samples_seq), weights_seq

        tau, samples_seq, weights_seq = sample_online(key, obs_seq, filt_state_seq, backwd_state_seq)

        return jnp.mean(tau), (samples_seq, weights_seq, filt_state_seq, backwd_state_seq)

class OnlineGeneralBackwardELBOSpecialInit:

    def __init__(self, p:HMM, q:BackwardSmoother, normalizer, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples
        self.normalizer = normalizer

    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        def q_filt_samples_and_log_probs(key, q_filt_state):
            samples = vmap(self.q.filt_dist.sample, in_axes=(0,None))(random.split(key, self.num_samples), q_filt_state.out)
            log_probs = vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples, q_filt_state.out)
            return samples, log_probs


        def init_functional(obs, log_prob, sample, new_obs, new_log_prob, new_sample, q_backwd_state):
            return self.p.transition_kernel.logpdf(new_sample, sample, theta.transition) \
                    + self.p.prior_dist.logpdf(sample, theta.prior) \
                    + self.p.emission_kernel.logpdf(new_obs, new_sample, theta.emission) \
                    + self.p.emission_kernel.logpdf(obs, sample, theta.emission) \
                    - self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_state) \
                    + log_prob \
                    - new_log_prob


        def additive_functional(obs, log_prob, sample, new_log_prob, new_sample, q_backwd_state):
            return self.p.transition_kernel.logpdf(new_sample, sample, theta.transition) \
                    + self.p.emission_kernel.logpdf(obs, new_sample, theta.emission) \
                    - self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_state) \
                    + log_prob \
                    - new_log_prob


        def init_tau(key, q_filt_states, q_backwd_state):

            key_filt, key_backwd = jax.random.split(key, 2)
            q_filt_state = tree_get_idx(0, q_filt_states)
            new_q_filt_state = tree_get_idx(1, q_filt_states)

            new_samples, new_log_probs = q_filt_samples_and_log_probs(key_filt, new_q_filt_state)

            backwd_sampler = vmap(self.q.backwd_kernel.sample, in_axes=(0,None,None))

            def init_component_tau(key, new_sample, new_log_prob):

                samples = backwd_sampler(jax.random.split(key, self.num_samples), new_sample, q_backwd_state)
                log_probs = vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples, q_filt_state.out)

                def sum_component(sample, log_prob):
                    return init_functional(obs_seq[0], log_prob, sample, obs_seq[1], new_log_prob, new_sample, q_backwd_state)

                return jnp.mean(vmap(sum_component)(samples, log_probs))

            tau = vmap(init_component_tau)(jax.random.split(key_backwd, self.num_samples), new_samples, new_log_probs)

            return tau, new_samples, new_log_probs

        def update_tau(carry, x):

            tau, samples, log_probs = carry 
            key, obs, q_filt_state, q_backwd_state = x 
            new_samples, new_log_probs = q_filt_samples_and_log_probs(key, q_filt_state)

            def update_component_tau(new_sample, new_log_prob):
                def sum_component(sample, log_prob, tau_component):
                    log_weight = self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_state) - log_prob
                    component = tau_component + additive_functional(obs, 
                                                                log_prob, 
                                                                sample, 
                                                                new_log_prob, 
                                                                new_sample, 
                                                                q_backwd_state)
                    return log_weight, component
                log_weights, components = vmap(sum_component)(samples, log_probs, tau)

                normalized_weights = self.normalizer(log_weights)
                return normalized_weights, jnp.sum(normalized_weights * components)
            
            weights, new_tau = vmap(update_component_tau)(new_samples, new_log_probs) 

            return (new_tau, new_samples, new_log_probs), (new_samples, weights)

        key, subkey = random.split(key, 2)

        filt_state_seq = self.q.compute_filt_state_seq(obs_seq, phi)
        backwd_state_seq = self.q.compute_kernel_state_seq(filt_state_seq, phi)

        tau, samples, log_probs = init_tau(subkey, 
                                        tree_get_slice(0, 2, filt_state_seq), 
                                        tree_get_idx(0, backwd_state_seq))           

        (tau, _ , _), (samples_seq, weights_seq) = lax.scan(f=update_tau, 
                                                            init=(tau, samples, log_probs), 
                                                            xs=(random.split(key, len(obs_seq)-2), 
                                                                obs_seq[2:],
                                                                tree_get_slice(2, None, filt_state_seq),
                                                                tree_get_slice(1, None, backwd_state_seq)))


        return jnp.mean(tau), (tree_prepend(samples, samples_seq), weights_seq, filt_state_seq, backwd_state_seq)

class OnlineGeneralBackwardELBONeo:

    def __init__(self, p:HMM, q:BackwardSmoother, normalizer, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples
        self.normalizer = normalizer

    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        q_filt_state_seq = self.q.compute_filt_state_seq(obs_seq, phi)
        q_backwd_state_seq = self.q.compute_kernel_state_seq(q_filt_state_seq, phi)

        backwd_sampler = vmap(self.q.backwd_kernel.sample, in_axes=(0,None,None))

        def q_filt_samples_and_log_probs(key, q_filt_state):
            samples = vmap(self.q.filt_dist.sample, in_axes=(0,None))(random.split(key, self.num_samples), q_filt_state.out)
            log_probs = vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples, q_filt_state.out)
            return samples, log_probs

        def additive_functional(obs, log_prob, sample, new_log_prob, new_sample, q_backwd_state):
            return self.p.transition_kernel.logpdf(new_sample, sample, theta.transition) \
                    + self.p.emission_kernel.logpdf(obs, new_sample, theta.emission) \
                    - self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_state) \
                    + log_prob \
                    - new_log_prob

        def init_functional(sample):
            return self.p.emission_kernel.logpdf(obs_seq[0], sample, theta.emission) \
                    + self.p.prior_dist.logpdf(sample, theta.prior)

        def update_tau(carry, x):

            tau_km1, samples_km1, log_probs_km1, q_filt_state_k, q_backwd_state_km1_k, obs_k = carry 
            key, obs_kp1, q_filt_state_kp1, q_backwd_state_k_kp1 = x 

            samples_kp1, log_probs_kp1 = q_filt_samples_and_log_probs(key, q_filt_state_kp1)

            def tau_kp1_component(key, sample_kp1, log_prob_kp1):

                samples_k_kp1 = backwd_sampler(random.split(key, self.num_backwd_samples), sample_kp1, q_backwd_state_k_kp1)
                log_probs_k_kp1 = vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples_k_kp1, q_filt_state_k)

                def tau_k_kp1_component(sample_k_kp1, log_prob_k_kp1):
                    def sum_component(sample_km1, log_prob_km1, tau_component_km1):
                        log_weight = self.q.backwd_kernel.logpdf(sample_km1, sample_k_kp1, q_backwd_state_km1_k) - log_prob_km1
                        tau_component_k_kp1 = tau_component_km1 + additive_functional(obs_k, 
                                                                    log_prob_km1, 
                                                                    sample_km1, 
                                                                    log_prob_k_kp1, 
                                                                    sample_k_kp1, 
                                                                    q_backwd_state_km1_k)
                        return log_weight, tau_component_k_kp1
                    log_weights, sum_components = vmap(sum_component)(samples_km1, log_probs_km1, tau_km1)

                    normalized_weights = self.normalizer(log_weights)
                    return jnp.sum(normalized_weights * sum_components)
            
                tau_k_kp1 = vmap(tau_k_kp1_component)(samples_k_kp1, log_probs_k_kp1) 

                def tau_kp1_sum_component(sample_k_kp1, log_prob_k_kp1, tau_component_k_kp1):
                    return tau_component_k_kp1 + additive_functional(obs_kp1, 
                                                                log_prob_k_kp1, 
                                                                sample_k_kp1, 
                                                                log_prob_kp1, 
                                                                sample_kp1, 
                                                                q_backwd_state_k_kp1)

                return jnp.mean(vmap(tau_kp1_sum_component)(samples_k_kp1, log_probs_k_kp1, tau_k_kp1))

            tau_kp1 = vmap(tau_kp1_component)(random.split(key, self.num_samples), samples_kp1, log_probs_kp1)
                               


            return (tau_kp1, samples_kp1, log_probs_kp1, q_filt_state_kp1, q_backwd_state_k_kp1), None

        key, subkey = random.split(key, 2)

        samples, log_probs = q_filt_samples_and_log_probs(subkey, tree_get_idx(0, q_filt_state_seq))

        tau = vmap(init_functional)(samples)                            

        (tau, _ , _), (samples_seq, weights_seq) = lax.scan(update_tau, 
                                                            init=(tau, samples, log_probs), 
                                                            xs=(random.split(key, len(obs_seq)-1), 
                                                                obs_seq[1:],
                                                                tree_dropfirst(q_filt_state_seq),
                                                                q_backwd_state_seq))



        return jnp.mean(tau), (tree_prepend(samples, samples_seq), weights_seq, filt_state_seq, backwd_state_seq)

class OnlineBackwardLinearELBO:

    def __init__(self, p:HMM, q:LinearBackwardSmoother, normalizer, num_samples=200):
        
        self.p = p
        self.q = q
        self.num_samples = num_samples
        self.normalizer = normalizer
            
    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        
        kl_term = quadratic_term_from_log_gaussian(theta.prior) #+ get_tractable_emission_term(obs_seq[0], theta.emission)

        q_filt_state = self.q.init_filt_state(obs_seq[0], phi)

        def V_step(carry, x):

            q_filt_state, kl_term = carry
            obs = x

            q_backwd_state = self.q.new_backwd_state(q_filt_state, phi)

            kl_term = expect_quadratic_term_under_backward(kl_term, q_backwd_state) \
                    + transition_term_integrated_under_backward(q_backwd_state, theta.transition)

            kl_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_state.noise.scale.log_det) +  0.5 * self.p.state_dim
            q_filt_state = self.q.new_filt_state(obs, q_filt_state, phi)


            return (q_filt_state, kl_term), (q_filt_state, q_backwd_state)
    
        (q_last_filt_state, kl_term), (q_filt_state_seq, q_backwd_state_seq) = lax.scan(V_step, 
                                                init=(q_filt_state, kl_term), 
                                                xs=obs_seq[1:])

        q_filt_state_seq = tree_prepend(q_filt_state, q_filt_state_seq)


        kl_term = expect_quadratic_term_under_gaussian(kl_term, q_last_filt_state.out) \
                - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_state.out.scale.log_det) \
                + 0.5*self.p.state_dim

        def sample_online(key, obs_seq, q_filt_state_seq, q_backwd_state_seq):

            def samples_and_log_probs(key, q_filt_state):
                samples = vmap(self.q.filt_dist.sample, in_axes=(0,None))(random.split(key, self.num_samples), q_filt_state.out)
                log_probs = vmap(self.q.filt_dist.logpdf, in_axes=(0,None))(samples, q_filt_state.out)
                return samples, log_probs

            key, subkey = random.split(key, 2)

            samples, log_probs = samples_and_log_probs(subkey, tree_get_idx(0, q_filt_state_seq))
            tau = vmap(self.p.emission_kernel.logpdf, in_axes=(None, 0, None))(obs_seq[0], samples, theta.emission)
            # tau = jnp.zeros(self.num_samples)
            def update_tau(carry, x):

                tau, samples, log_probs = carry 
                key, obs, q_filt_state, q_backwd_state = x 
                new_samples, new_log_probs = samples_and_log_probs(key, q_filt_state)

                def update_component_tau(new_sample):
                    def sum_component(sample, log_prob, tau_component):
                        log_weight = self.q.backwd_kernel.logpdf(sample, new_sample, q_backwd_state) - log_prob
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
                                tree_dropfirst(q_filt_state_seq),
                                q_backwd_state_seq))

        (tau, _, _), log_weights = sample_online(key, obs_seq, q_filt_state_seq, q_backwd_state_seq)

        return kl_term + jnp.mean(tau), log_weights

class BackwardLinearELBO:

    def __init__(self, p:HMM, q:LinearBackwardSmoother, num_samples=200):
        
        self.p = p
        self.q = q
        self.num_samples = num_samples
            
    def __call__(self, key, obs_seq, theta:HMMParams, phi):

        
        kl_term = quadratic_term_from_log_gaussian(theta.prior) #+ get_tractable_emission_term(obs_seq[0], theta.emission)

        q_filt_state = self.q.init_filt_state(obs_seq[0], phi)

        def V_step(carry, x):

            q_filt_state, kl_term = carry
            obs = x

            q_backwd_state = self.q.new_backwd_state(q_filt_state, phi)

            kl_term = expect_quadratic_term_under_backward(kl_term, q_backwd_state) \
                    + transition_term_integrated_under_backward(q_backwd_state, theta.transition)

            kl_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_state.noise.scale.log_det) +  0.5 * self.p.state_dim
            q_filt_state = self.q.new_filt_state(obs, q_filt_state, phi)


            return (q_filt_state, kl_term), q_backwd_state
    
        (q_last_filt_state, kl_term), q_backwd_state_seq = lax.scan(V_step, 
                                                init=(q_filt_state, kl_term), 
                                                xs=obs_seq[1:])


        kl_term =  expect_quadratic_term_under_gaussian(kl_term, q_last_filt_state.out) \
                    - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_state.out.scale.log_det) \
                    + 0.5*self.p.state_dim


        marginals = self.q.compute_marginals(q_last_filt_state, q_backwd_state_seq)

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


        q_filt_state = self.q.init_filt_state(obs_seq[0], phi)

        def V_step(state, obs):

            q_filt_state, kl_term = state
            q_backwd_state = self.q.new_backwd_state(q_filt_state, phi)

            kl_term = expect_quadratic_term_under_backward(kl_term, q_backwd_state) \
                    + transition_term_integrated_under_backward(q_backwd_state, theta.transition) \
                    + get_tractable_emission_term(obs, theta.emission)


            kl_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_state.noise.scale.log_det) +  0.5 * self.p.state_dim
            q_filt_state = self.q.new_filt_state(obs, q_filt_state, phi)

            return (q_filt_state, kl_term), q_backwd_state
    
        (q_last_filt_state, result) = lax.scan(V_step, 
                                                init=(q_filt_state, result), 
                                                xs=obs_seq[1:])[0]


        return expect_quadratic_term_under_gaussian(result, q_last_filt_state.out) \
                    - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_state.out.scale.log_det) \
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
                online=False):

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


        def batch_step(carry, x):

            def step(params, opt_state, batch, keys):
                # neg_elbo_values = jax.vmap(self.loss, in_axes=(0,0,None))(keys, batch, params)

                neg_elbo_values, grads = jax.vmap(jax.value_and_grad(self.loss, argnums=2), in_axes=(0,0,None))(keys, batch, params)
                avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), grads)
                
                updates, opt_state = self.optimizer.update(avg_grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, \
                    opt_state, \
                    -jnp.mean(neg_elbo_values / seq_length), \
                    tree_map(lambda x,y: x if y else jnp.zeros_like(x), avg_grads, self.trainable_params)

            data, params, opt_state, subkeys_epoch = carry
            batch_start = x
            batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
            batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)
            params, opt_state, avg_elbo_batch, avg_grads_batch = step(params, opt_state, batch_obs_seq, batch_keys)
            return (data, params, opt_state, subkeys_epoch), (avg_elbo_batch, avg_grads_batch)


        avg_elbos = []
        all_params = []
        batch_start_indices = jnp.arange(0, num_seqs, self.batch_size)

        t = tqdm(total=self.num_epochs, desc='Epoch')
        for epoch_nb in range(self.num_epochs):
            t.update(1)
            subkeys_epoch = subkeys[epoch_nb]
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            
            data = jax.random.permutation(subkey_batcher, data)
        

            # print(self.loss(key_batcher, data[0], params))
            (_ , params, opt_state, _), (avg_elbo_batches, avg_grads_batches) = jax.lax.scan(batch_step,  
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


