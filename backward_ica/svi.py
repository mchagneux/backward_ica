import jax
import optax
from jax import vmap, lax, config, numpy as jnp
from jax.random import normal
config.update("jax_enable_x64", True)

from .hmm import *
from .utils import *


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

    A = transition_params.matrix @ q_backwd_state.matrix - jnp.eye(transition_params.cov.shape[0])
    b = transition_params.matrix @ q_backwd_state.bias + transition_params.bias
    Omega = transition_params.prec
    
    result = -0.5 * QuadTerm.from_A_b_Omega(A, b, Omega)
    result.c += -0.5 * jnp.trace(transition_params.prec @ transition_params.matrix @ q_backwd_state.cov @ transition_params.matrix.T) \
                + constant_terms_from_log_gaussian(transition_params.cov.shape[0], transition_params.log_det)
    return result 

def expect_quadratic_term_under_backward(quad_form:QuadTerm, backwd_state):
    # the result is still a quadratic forms with new parameters, following the formula for expected values of quadratic forms  

    W = backwd_state.matrix.T @ quad_form.W @ backwd_state.matrix
    v = backwd_state.matrix.T @ (quad_form.v + (quad_form.W + quad_form.W.T) @ backwd_state.bias)
    c = quad_form.c + jnp.trace(quad_form.W @ backwd_state.cov) + backwd_state.bias.T @ quad_form.W @ backwd_state.bias + quad_form.v.T @ backwd_state.bias 

    return QuadTerm(W=W, v=v, c=c)

def expect_quadratic_term_under_gaussian(quad_form:QuadTerm, gaussian_params):
    return jnp.trace(quad_form.W @ gaussian_params.cov) + quad_form.evaluate(gaussian_params.mean)

def quadratic_term_from_log_gaussian(gaussian_params):

    result = - 0.5 * QuadTerm(W=gaussian_params.prec, 
                    v=-(gaussian_params.prec + gaussian_params.prec.T) @ gaussian_params.mean, 
                    c=gaussian_params.mean.T @ gaussian_params.prec @ gaussian_params.mean)

    result.c += constant_terms_from_log_gaussian(gaussian_params.cov.shape[0], gaussian_params.log_det)

    return result

class ELBO:

    def __init__(self, p:GaussianHMM, q:Smoother, num_samples=1):
        self.p = p
        self.q = q
        self.num_samples = num_samples

    def V_step(self, state, obs):

        q_filt_state, tractable_term, p_params, q_params = state
        q_backwd_state = self.q.new_backwd_state(q_filt_state, q_params)

        tractable_term = expect_quadratic_term_under_backward(tractable_term, q_backwd_state) \
                + transition_term_integrated_under_backward(q_backwd_state, p_params.transition)


        tractable_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, q_backwd_state.log_det) +  0.5 * self.p.state_dim
        q_filt_state = self.q.new_filt_state(obs, q_filt_state, q_params)

        return (q_filt_state, tractable_term, p_params, q_params), q_backwd_state

    def compute_tractable_terms(self, obs_seq, p_params, q_params):
        tractable_term = quadratic_term_from_log_gaussian(p_params.prior)

        q_filt_state = self.q.init_filt_state(obs_seq[0], p_params.prior, q_params)
    
        (q_last_filt_state, tractable_term, p_params, q_params), q_backwd_state_seq = lax.scan(self.V_step, 
                                                        init=(q_filt_state, tractable_term, p_params, q_params), 
                                                        xs=obs_seq[1:])



        tractable_term = expect_quadratic_term_under_gaussian(tractable_term, q_last_filt_state) \
                    - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_state.log_det) \
                    + 0.5*self.p.state_dim

        return tractable_term, (q_last_filt_state, q_backwd_state_seq)
        
    def compute(self, obs_seq, key, p_params, q_params):

        tractable_term, (q_last_filt_state, q_backwd_state_seq) = self.compute_tractable_terms(obs_seq, p_params, q_params)
        marginal_means, marginal_covs = self.q.backwd_pass(q_last_filt_state, q_backwd_state_seq)

        def exact_expectation(marginal_mean, marginal_cov, obs, p_params):
            A = p_params.emission.matrix
            b = p_params.emission.bias - obs
            Omega = p_params.emission.prec
            gaussian_params = GaussianParams(mean=marginal_mean, cov_chol=None, cov=marginal_cov, prec=None, log_det=None)
            return expect_quadratic_term_under_gaussian(-0.5*QuadTerm.from_A_b_Omega(A, b, Omega), gaussian_params)

        def monte_carlo_sample(normal_sample, marginal_mean, marginal_cov_chol, obs, p):
            common_term = obs - p.emission.map(marginal_mean + marginal_cov_chol @ normal_sample)
            return -0.5 * (common_term.T @ p.emission.prec @ common_term)

        # normal_samples = normal(key, shape=(self.num_samples, *marginal_means.shape))
        # marginal_covs_chol = jnp.linalg.cholesky(marginal_covs)
        # monte_carlo_samples = vmap(vmap(monte_carlo_sample, in_axes=(0,0,0,0,None)), in_axes=(0,None,None,None,None))(normal_samples, marginal_means, marginal_covs_chol, obs_seq, p)
        # monte_carlo_term = jnp.sum(jnp.mean(monte_carlo_samples, axis=0))
        
        monte_carlo_term = jnp.sum(vmap(exact_expectation, in_axes=(0,0,0,None))(marginal_means, marginal_covs, obs_seq, p_params))
        
        monte_carlo_term += obs_seq.shape[0] * constant_terms_from_log_gaussian(p_params.emission.cov.shape[0], p_params.emission.log_det)

                        
        return monte_carlo_term + tractable_term
        
class SVI:

    def __init__(self, p:GaussianHMM, q:Smoother, optimizer, num_epochs, batch_size, num_samples=1):

        self.optimizer = optimizer 
        self.num_samples = num_samples
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.q = q 
        self.p = p 
        self.loss = lambda seq, key, p_params, q_params: -ELBO(self.p, self.q, self.num_samples).compute(seq, key, p_params, q_params)

    def fit(self, data, p_params, q_params, subkey_montecarlo=None):

        loss = lambda seq, key, q_params:self.loss(seq, key, self.p.format_params(p_params), self.q.format_params(q_params))
        
        opt_state = self.optimizer.init(q_params)
        num_seqs = data.shape[0]
        subkeys = jnp.empty((self.num_epochs, num_seqs, 1))
        
        # subkeys = jax.random.split(subkey_montecarlo, num_seqs * num_epochs)
        # subkeys = jnp.array(subkeys).reshape(num_epochs,num_seqs,-1)

        @jax.jit
        def batch_step(carry, x):

            def q_step(q_params, opt_state, batch, keys):
                neg_elbo_values, grads = jax.vmap(jax.value_and_grad(loss, argnums=2), in_axes=(0,0,None))(batch, keys, q_params)
                avg_grads = jax.tree_util.tree_map(jnp.mean, grads)
                updates, opt_state = self.optimizer.update(avg_grads, opt_state, q_params)
                q_params = optax.apply_updates(q_params, updates)
                return q_params, opt_state, jnp.mean(-neg_elbo_values)

            q_params, opt_state, subkeys_epoch = carry
            batch_start = x
            batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
            batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)
            q_params, opt_state, avg_elbo_batch = q_step(q_params, opt_state, batch_obs_seq, batch_keys)
            return (q_params, opt_state, subkeys_epoch), avg_elbo_batch

        def epoch_step(carry, x):
            q_params, opt_state = carry
            subkeys_epoch = x
            batch_start_indices = jnp.arange(0, num_seqs, self.batch_size)
        

            (q_params, opt_state, _), avg_elbo_batches = jax.lax.scan(batch_step, 
                                                                init=(q_params, opt_state, subkeys_epoch), 
                                                                xs = batch_start_indices)

            return (q_params, opt_state), jnp.mean(avg_elbo_batches)


        (q_params, _), avg_elbos = jax.lax.scan(epoch_step, init=(q_params, opt_state), xs=subkeys)
        
        return q_params, avg_elbos

    def multi_fit(self, data, p_params, key, num_fits=1):
        all_avg_elbos = []
        all_fitted_params = []
        for key in jax.random.split(key, num_fits):
            fitted_params, avg_elbos = self.fit(data, p_params, self.q.get_random_params(key))
            all_avg_elbos.append(avg_elbos)
            all_fitted_params.append(fitted_params)
        best_optim = jnp.argmax(jnp.array([avg_elbos[-1] for avg_elbos in all_avg_elbos]))
        return all_fitted_params[best_optim], all_avg_elbos[best_optim]

def check_linear_gaussian_elbo(data, p:LinearGaussianHMM, p_params):
    evidence_via_elbo_on_seq = lambda seq: ELBO(p,p).compute(seq, None, p.format_params(p_params), p.format_params(p_params))
    evidence_via_kalman_on_seq = lambda seq: p.likelihood_seq(seq, p_params)
    print('ELBO sanity check:',jnp.abs(jnp.mean(jax.vmap(evidence_via_elbo_on_seq)(data) - jax.vmap(evidence_via_kalman_on_seq)(data))))

