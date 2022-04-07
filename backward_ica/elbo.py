from dataclasses import dataclass
from typing import * 
from jax import vmap, lax, config, numpy as jnp
from jax.tree_util import register_pytree_node_class
from .hmm import *
from typing import * 
config.update("jax_enable_x64", True)
from jax.random import normal
# config.update("jax_check_tracer_leaks", True)
config.update("jax_debug_nans", True)

### Some abstractions for frequently used objects when computing elbo via backwards decomposition

@dataclass(init=True)
@register_pytree_node_class
class QuadTerm:

    W: jnp.ndarray
    v: jnp.ndarray
    c: jnp.ndarray

    def __iter__(self):
        return iter((self.W, self.v, self.c))

    def __add__(self, other):
        return QuadTerm(W = self.W + other.W, 
                        v = self.v + other.v, 
                        c = self.c + other.c)

    def __rmul__(self, other):
        return QuadTerm(W=other*self.W, 
                        v=other*self.v, 
                        c=other*self.c) 

    def evaluate(self, x):
        return x.T @ self.W @ x + self.v.T @ x + self.c

    def tree_flatten(self):
        return ((self.W, self.v, self.c), None) 

    @staticmethod
    def from_A_b_Omega(A, b, Omega):
        return QuadTerm(W = A.T @ Omega @ A, 
                        v = A.T @ (Omega + Omega.T) @ b, 
                        c = b.T @ Omega @ b)
    @staticmethod 
    def evaluate_from_A_b_Omega(A, b, Omega, x):
        common_term = A @ x + b 
        return common_term.T @ Omega @ common_term



    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
    
def constant_terms_from_log_gaussian(dim:int, det:float)->float:
    """Utility function to compute the log of the term that is against the exponential for a multivariate Normal

    Args:
        dim (int): the dimension of the support of the multivariate Normal
        det (float): the precomputed determinant of the covariance matrix 

    Returns:
        float: the value of the requested factor  
    """

    return -0.5*(dim * jnp.log(2*jnp.pi) + jnp.log(det))

def transition_term_integrated_under_backward(q_backwd_state, transition_params):
    # expectation of the quadratic form that appears in the log of the state transition density

    A = transition_params.matrix @ q_backwd_state.matrix - jnp.eye(transition_params.cov.shape[0])
    b = transition_params.matrix @ q_backwd_state.bias + transition_params.bias
    Omega = transition_params.prec
    
    result = -0.5 * QuadTerm.from_A_b_Omega(A, b, Omega)
    result.c += -0.5 * jnp.trace(transition_params.prec @ transition_params.matrix @ q_backwd_state.cov @ transition_params.matrix.T) \
                + constant_terms_from_log_gaussian(transition_params.cov.shape[0], transition_params.det)
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

    result.c += constant_terms_from_log_gaussian(gaussian_params.cov.shape[0], gaussian_params.det)

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


        tractable_term.c += -constant_terms_from_log_gaussian(self.p.state_dim, jnp.linalg.det(q_backwd_state.cov)) +  0.5 * self.p.state_dim
        q_filt_state = self.q.new_filt_state(obs, q_filt_state, q_params)
        return (q_filt_state, tractable_term, p_params, q_params), q_backwd_state

    def compute_tractable_terms(self, obs_seq, p_params, q_params):
        tractable_term = quadratic_term_from_log_gaussian(p_params.prior)

        q_filt_state = self.q.init_filt_state(obs_seq[0], p_params.prior, q_params)
        

        # #--- debugging
        # q_backward_seq = []
        # for obs in obs_seq[1:]:
        #     (q_filtering, tractable_term, p, q_params), q_backward = self.V_step((q_filtering, tractable_term, p, q_params), obs)
        #     q_backward_seq.append(q_backward)
        # weights = jnp.stack(tuple(q_backward.matrix for q_backward in q_backward_seq))
        # biases = jnp.stack(tuple(q_backward.bias for q_backward in q_backward_seq))
        # covs = jnp.stack(tuple(q_backward.cov for q_backward in q_backward_seq))
        # q_backward_seq = LinearGaussianKernel(_mappings['linear'],
        #                                     {'weight':weights,'bias':biases},
        #                                     covs)
        # #---


        (q_last_filt_state, tractable_term, p_params, q_params), q_backwd_state_seq = lax.scan(self.V_step, 
                                                        init=(q_filt_state, tractable_term, p_params, q_params), 
                                                        xs=obs_seq[1:])



        tractable_term = expect_quadratic_term_under_gaussian(tractable_term, q_last_filt_state) \
                    - constant_terms_from_log_gaussian(self.p.state_dim, q_last_filt_state.det) \
                    + 0.5*self.p.state_dim

        return tractable_term, (q_last_filt_state, q_backwd_state_seq)
        
    def compute(self, obs_seq, key, p_params, q_params):

        p_params = self.p.format_params(p_params)
        q_params = self.q.format_params(q_params)

        tractable_term, (q_last_filt_state, q_backwd_state_seq) = self.compute_tractable_terms(obs_seq, p_params, q_params)
        marginal_means, marginal_covs = self.q.backwd_pass(q_last_filt_state, q_backwd_state_seq)

        def exact_expectation(marginal_mean, marginal_cov, obs, p_params):
            A = p_params.emission.matrix
            b = p_params.emission.bias - obs
            Omega = p_params.emission.prec
            gaussian_params = GaussianParams(mean=marginal_mean, cov_base=None, cov=marginal_cov, prec=None, det=None)
            return expect_quadratic_term_under_gaussian(-0.5*QuadTerm.from_A_b_Omega(A, b, Omega), gaussian_params)

        def monte_carlo_sample(normal_sample, marginal_mean, marginal_cov_chol, obs, p):
            common_term = obs - p.emission.map(marginal_mean + marginal_cov_chol @ normal_sample)
            return -0.5 * (common_term.T @ p.emission.prec @ common_term)

        # normal_samples = normal(key, shape=(self.num_samples, *marginal_means.shape))
        # marginal_covs_chol = jnp.linalg.cholesky(marginal_covs)
        # monte_carlo_samples = vmap(vmap(monte_carlo_sample, in_axes=(0,0,0,0,None)), in_axes=(0,None,None,None,None))(normal_samples, marginal_means, marginal_covs_chol, obs_seq, p)
        # monte_carlo_term = jnp.sum(jnp.mean(monte_carlo_samples, axis=0))
        
        monte_carlo_term = jnp.sum(vmap(exact_expectation, in_axes=(0,0,0,None))(marginal_means, marginal_covs, obs_seq, p_params))
        
        monte_carlo_term += obs_seq.shape[0] * constant_terms_from_log_gaussian(p_params.emission.cov.shape[0], p_params.emission.det)

                        
        return -(monte_carlo_term + tractable_term)
        






         





        
        

            



    


# def get_neg_elbo(p_model, q_model, aux_defs=None):

#     if p_model['transition']['mapping_type'] == 'linear':
#         if p_model['emission']['mapping_type'] == 'linear':
#             return LinearELBO(p_model, q_model).compute
#         elif p_model['emission']['mapping_type'] == 'nonlinear': 
#             if aux_defs is None: raise NotImplementedError
#             return NonLinearELBO(p_model, q_model, aux_defs).compute
    # else: 
    #     raise NotImplementedError
 