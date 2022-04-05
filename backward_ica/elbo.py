from dataclasses import dataclass
from turtle import back, backward
from typing import * 
from jax import vmap, lax, config, numpy as jnp
from jax.tree_util import register_pytree_node_class, Partial
from .hmm import GaussianHMM, update_backward as hmm_backward_update

from .kalman import filter_step as kalman_filter_step, init as kalman_init, predict as kalman_predict, update as kalman_update
from .utils import GaussianKernel, _mappings, LinearGaussianKernel, Gaussian, prec_and_det, LinearGaussianKernel
from typing import * 
from abc import ABCMeta, abstractmethod
from functools import partial
config.update("jax_enable_x64", True)
import numpy as np 
from jax.random import normal
# config.update("jax_check_tracer_leaks", True)

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
    
def constant_terms_from_log_gaussian(dim:int, det_cov:float)->float:
    """Utility function to compute the log of the term that is against the exponential for a multivariate Normal

    Args:
        dim (int): the dimension of the support of the multivariate Normal
        det_cov (float): the precomputed determinant of the covariance matrix 

    Returns:
        float: the value of the requested factor  
    """

    return -0.5*(dim * jnp.log(2*jnp.pi) + jnp.log(det_cov))

def transition_term_integrated_under_backward(q_backward:LinearGaussianKernel, p_transition:LinearGaussianKernel):
    # expectation of the quadratic form that appears in the log of the state transition density

    A = p_transition.weight @ q_backward.weight - jnp.eye(p_transition.cov.shape[0])
    b = p_transition.weight @ q_backward.bias + p_transition.bias
    Omega = p_transition.prec
    
    result = -0.5 * QuadTerm.from_A_b_Omega(A, b, Omega)
    result.c += -0.5 * jnp.trace(p_transition.prec @ p_transition.weight @ q_backward.cov @ p_transition.weight.T) \
                + constant_terms_from_log_gaussian(p_transition.cov.shape[0], p_transition.det_cov)
    return result 

def expect_quadratic_term_under_backward(quad_form:QuadTerm, q_backward:LinearGaussianKernel):
    # the result is still a quadratic forms with new parameters, following the formula for expected values of quadratic forms  

    W = q_backward.weight.T @ quad_form.W @ q_backward.weight
    v = q_backward.weight.T @ (quad_form.v + (quad_form.W + quad_form.W.T) @ q_backward.bias)
    c = quad_form.c + jnp.trace(quad_form.W @ q_backward.cov) + q_backward.bias.T @ quad_form.W @ q_backward.bias + quad_form.v.T @ q_backward.bias 

    return QuadTerm(W=W, v=v, c=c)

def expect_quadratic_term_under_filtering(quad_form:QuadTerm, q_filtering:Gaussian):
    return jnp.trace(quad_form.W @ q_filtering.cov) + quad_form.evaluate(q_filtering.mean)

def quadratic_term_from_log_gaussian(gaussian:Gaussian):

    result = - 0.5 * QuadTerm(W=gaussian.prec, 
                    v=-(gaussian.prec + gaussian.prec.T) @ gaussian.mean, 
                    c=gaussian.mean.T @ gaussian.prec @ gaussian.mean)

    result.c += constant_terms_from_log_gaussian(gaussian.cov.shape[0], gaussian.det_cov)

    return result

class Q(metaclass=ABCMeta):
    def __init__(self, q_model):
        self.model = q_model  

    def format_params(self, params):
        return params  
        
    @abstractmethod
    def update_filtering(self, obs, q_filtering, q_params):
        raise NotImplementedError
        
    @abstractmethod
    def update_backward(self, q_filtering, q_params):
        raise NotImplementedError

    def marginals_from_filtering_and_backward(self, q_filtering, q_backward_seq):
        def step(next_filt_mean_cov, q_backward):
            next_filt_mean, next_filt_cov = next_filt_mean_cov
            backwd_A, backwd_a, backwd_cov = q_backward.weight, q_backward.bias, q_backward.cov
            mean = backwd_A @ next_filt_mean + backwd_a
            cov = backwd_A @ next_filt_cov @ backwd_A.T + backwd_cov
            return (mean, cov), (mean, cov)

        _, (means, covs) = lax.scan(step, init=(q_filtering.mean, q_filtering.cov), xs=q_backward_seq, reverse=True)

        means = jnp.concatenate([means, q_filtering.mean[None,:]])
        covs = jnp.concatenate([covs, q_filtering.cov[None,:]])
        return means, covs 



class QFromForward(Q):
    def __init__(self, q_model):
        super().__init__(q_model)

    def init_filtering(self, obs, q_params, p):
        mean, cov = kalman_init(obs, q_params.prior, q_params.emission)[2:]
        return Gaussian(mean, cov, *prec_and_det(cov))

    def format_params(self, params):
        return GaussianHMM.build_from_dict(params, self.model)

    def update_filtering(self, obs, q_filtering, q_params):
        mean, cov = kalman_filter_step(q_filtering.mean, q_filtering.cov, obs, q_params.transition, q_params.emission)[2:]
        return Gaussian(mean, cov, *prec_and_det(cov))

    def update_backward(self, q_filtering, q_params):
        A, a, cov, prec = hmm_backward_update(q_filtering, q_params)

        return LinearGaussianKernel(mapping=_mappings['linear'], 
                            mapping_params={'weight':A, 'bias':a},
                            cov=cov,
                            prec=prec, 
                            det_cov=jnp.linalg.det(cov))

    def marginals(self, obs_seq, q_params, p):
        q_params = GaussianHMM.build_from_dict(q_params, self.model)
        q_filtering = self.init_filtering(obs_seq[0], q_params, p)

        def forward_step(q_filtering, obs):
            q_backward = self.update_backward(q_filtering, q_params)
            q_filtering = self.update_filtering(obs, q_filtering, q_params)
            return q_filtering, q_backward
        

        q_filtering, q_backward_seq = lax.scan(forward_step, 
                                        init=q_filtering,
                                        xs=obs_seq[1:])

        return self.marginals_from_filtering_and_backward(q_filtering, q_backward_seq)

class QFromBackward(Q):
    def __init__(self, q_model):
        super().__init__(q_model)

    def init_filtering(self, obs, q_params, p):
        mean, cov = kalman_init(obs, p.prior, p.emission)[2:]
        return Gaussian(mean, cov, *prec_and_det(cov))

    def update_filtering(self, obs, q_filtering:Gaussian, q_params):

        mean, cov = self.model['filtering']['update'](obs=obs,
                                                    filt_mean=q_filtering.mean,
                                                    filt_cov=q_filtering.cov,
                                                    params=q_params['filtering']['update'])

        return Gaussian(mean, cov, *prec_and_det(cov))

    def update_backward(self, q_filtering:Gaussian, q_params):

        A, a, cov = self.model['backward'](filt_mean=q_filtering.mean, 
                                        filt_cov=q_filtering.cov, 
                                        params=q_params['backward'])

        return LinearGaussianKernel(_mappings['linear'], 
                                    {'weight':A, 'bias':a},
                                    cov,
                                    *prec_and_det(cov))
    def marginals(self, obs_seq, q_params, p):
        q_filtering = self.init_filtering(obs_seq[0], q_params, p)

        def forward_step(q_filtering, obs):
            q_backward = self.update_backward(q_filtering, q_params)
            q_filtering = self.update_filtering(obs, q_filtering, q_params)
            return q_filtering, q_backward
        

        q_filtering, q_backward_seq = lax.scan(forward_step, 
                                        init=q_filtering,
                                        xs=obs_seq[1:])

        return self.marginals_from_filtering_and_backward(q_filtering, q_backward_seq)

class NonLinearELBO:

    def __init__(self, p_model, q:Q, num_samples=1):
        self.p_model = p_model
        self.q = q
        self.num_samples = num_samples

    def V_step(self, state, obs):
        q_filtering, tractable_term, p, q_params = state
        q_backward = self.q.update_backward(q_filtering, q_params)
        tractable_term = expect_quadratic_term_under_backward(tractable_term, q_backward) \
                + transition_term_integrated_under_backward(q_backward, p.transition)


        dim_z = p.transition.cov.shape[0]
        tractable_term.c += -constant_terms_from_log_gaussian(dim_z, jnp.linalg.det(q_backward.cov)) +  0.5 * dim_z
        q_filtering = self.q.update_filtering(obs, q_filtering, q_params)
        return (q_filtering, tractable_term, p, q_params), q_backward

    def compute_tractable_terms(self, obs_seq, p, q_params):
        tractable_term = quadratic_term_from_log_gaussian(p.prior)

        q_filtering = self.q.init_filtering(obs_seq[0], q_params, p)
        

        (q_filtering, tractable_term, p, q_params), q_backward_seq = lax.scan(self.V_step, 
                                                        init=(q_filtering, tractable_term, p, q_params), 
                                                        xs=obs_seq[1:])

        tractable_term = expect_quadratic_term_under_filtering(tractable_term, q_filtering) \
                    - constant_terms_from_log_gaussian(q_filtering.cov.shape[0], q_filtering.det_cov) \
                    + 0.5*q_filtering.cov.shape[0]


        return tractable_term, (q_filtering, q_backward_seq)
        
    def compute(self, obs_seq, key, p_params, q_params):

        p = GaussianHMM.build_from_dict(p_params, self.p_model)
        q_params = self.q.format_params(q_params)

        tractable_term, (q_filtering, q_backward_seq) = self.compute_tractable_terms(obs_seq, p, q_params)
        marginal_means, marginal_covs = self.q.marginals_from_filtering_and_backward(q_filtering, q_backward_seq)

        
        def exact_expectation(marginal_mean, marginal_cov, obs, p):
            p_emission = p.emission
            A = p_emission.weight
            b = p_emission.bias - obs
            Omega = p_emission.prec
            return expect_quadratic_term_under_filtering(-0.5*QuadTerm.from_A_b_Omega(A, b, Omega), Gaussian(marginal_mean, marginal_cov, *prec_and_det(marginal_cov)))

        def monte_carlo_sample(normal_sample, marginal_mean, marginal_cov_chol, obs, p):
            common_term = obs - p.emission.map(marginal_mean + marginal_cov_chol @ normal_sample)
            return -0.5 * (common_term.T @ p.emission.prec @ common_term)

        # normal_samples = normal(key, shape=(self.num_samples, *marginal_means.shape))
        # marginal_covs_chol = jnp.linalg.cholesky(marginal_covs)
        # monte_carlo_samples = vmap(vmap(monte_carlo_sample, in_axes=(0,0,0,0,None)), in_axes=(0,None,None,None,None))(normal_samples, marginal_means, marginal_covs_chol, obs_seq, p)
        # monte_carlo_term = jnp.sum(jnp.mean(monte_carlo_samples, axis=0))
        
        monte_carlo_term = jnp.sum(vmap(exact_expectation, in_axes=(0,0,0,None))(marginal_means, marginal_covs, obs_seq, p))
        
        monte_carlo_term += obs_seq.shape[0] * constant_terms_from_log_gaussian(p.emission.cov.shape[0], p.emission.det_cov)

                        
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
 