from dataclasses import dataclass
from typing import * 
from jax import vmap, lax, config, numpy as jnp
from jax.tree_util import register_pytree_node_class, Partial
from .hmm import GaussianHMM

from .kalman import filter_step as kalman_filter_step, init as kalman_init
from .utils import *
from typing import * 
from abc import ABCMeta, abstractmethod
from functools import partial
config.update("jax_enable_x64", True)

### Some abstractions for frequently used objects when computing elbo via backwards decomposition

@dataclass(init=True)
@register_pytree_node_class
class QuadForm:

    W: jnp.ndarray
    v: jnp.ndarray
    c: jnp.ndarray

    def __iter__(self):
        return iter((self.W, self.v, self.c))

    def __add__(self, other):
        return QuadForm(W = self.W + other.W, 
                        v = self.v + other.v, 
                        c = self.c + other.c)

    def __rmul__(self, other):
        return QuadForm(W=other*self.W, 
                        v=other*self.v, 
                        c=other*self.c) 

    def evaluate(self, x):
        return x.T @ self.W @ x + self.v.T @ x + self.c

    def tree_flatten(self):
        return ((self.W, self.v, self.c), None) 

    @staticmethod
    def from_A_b_Omega(A, b, Omega):
        return QuadForm(W = A.T @ Omega @ A, 
                        v = A.T @ (Omega + Omega.T) @ b, 
                        c = b.T @ Omega @ b)
    @staticmethod 
    def evaluate_from_A_b_Omega(A, b, Omega, x):
        common_term = A @ x + b 
        return common_term.T @ Omega @ common_term



    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    
@dataclass(init=True)
@register_pytree_node_class
class FilteringParams:

    mean:jnp.ndarray
    cov:jnp.ndarray

    def __iter__(self):
        return iter((self.mean, self.cov))
    
    def tree_flatten(self):
        return ((self.mean, self.cov), None) 

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@dataclass(init=True)
@register_pytree_node_class
class BackwardParams:

    A:jnp.ndarray
    a:jnp.ndarray
    cov:jnp.ndarray

    def __iter__(self):
        return iter((self.A, self.a, self.cov))
    
    def tree_flatten(self):
        return ((self.A, self.a, self.cov), None) 

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

def init_filtering(observation, q_prior:Gaussian, q_emission:GaussianKernel):
    return FilteringParams(*kalman_init(observation, q_prior, q_emission)[2:])

def update_filtering(observation, q_filtering:FilteringParams, q_transition:GaussianKernel, q_emission:GaussianKernel):
    return FilteringParams(*kalman_filter_step(*q_filtering, observation, q_transition, q_emission)[2:])

def update_backward(q_filtering:FilteringParams, q_transition:GaussianKernel):

    filtering_prec = jnp.linalg.inv(q_filtering.cov)

    backward_prec = q_transition.weight.T @ q_transition.prec @ q_transition.weight + filtering_prec

    cov = jnp.linalg.inv(backward_prec)

    common_term = q_transition.weight.T @ q_transition.prec
    A = cov @ common_term
    a = cov @ (filtering_prec @ q_filtering.mean - common_term @  q_transition.bias)

    return BackwardParams(A,a,cov)

def transition_term_integrated_under_backward(q_backward:BackwardParams, p_transition:GaussianKernel):
    # expectation of the quadratic form that appears in the log of the state transition density

    A = p_transition.weight @ q_backward.A - jnp.eye(p_transition.cov.shape[0])
    b = p_transition.weight @ q_backward.a + p_transition.bias
    Omega = p_transition.prec
    
    result = -0.5 * QuadForm.from_A_b_Omega(A, b, Omega)
    result.c += -0.5 * jnp.trace(p_transition.prec @ p_transition.weight @ q_backward.cov @ p_transition.weight.T) \
                + constant_terms_from_log_gaussian(p_transition.cov.shape[0], p_transition.det_cov)
    return result 

def expect_quadratic_term_under_backward(quad_form:QuadForm, q_backward:BackwardParams):
    # the result is still a quadratic forms with new parameters, following the formula for expected values of quadratic forms  

    W = q_backward.A.T @ quad_form.W @ q_backward.A
    v = q_backward.A.T @ (quad_form.v + (quad_form.W + quad_form.W.T) @ q_backward.a)
    c = quad_form.c + jnp.trace(quad_form.W @ q_backward.cov) + q_backward.a.T @ quad_form.W @ q_backward.a + quad_form.v.T @ q_backward.a 

    return QuadForm(W=W, v=v, c=c)

def expect_quadratic_term_under_filtering(quad_form:QuadForm, q_filtering:FilteringParams):
    return jnp.trace(quad_form.W @ q_filtering.cov) + quad_form.evaluate(q_filtering.mean)

def quadratic_term_from_log_gaussian(gaussian:Gaussian):

    result = - 0.5 * QuadForm(W=gaussian.prec, 
                    v=-(gaussian.prec + gaussian.prec.T) @ gaussian.mean, 
                    c=gaussian.mean.T @ gaussian.prec @ gaussian.mean)

    result.c += constant_terms_from_log_gaussian(gaussian.cov.shape[0], gaussian.det_cov)

    return result

class BackwardELBO(metaclass=ABCMeta): 

    def __init__(self, p_def, q_def):
        self.p_def = p_def 
        self.q_def = q_def 

    def set_aux(self, aux_params):
        return None 

    @abstractmethod
    def _get_obs_term(self, *args):
        raise NotImplementedError 

    @abstractmethod
    def _init_V(self, *args):
        raise NotImplementedError
    
    @abstractmethod
    def _update_V(self, *args):
        raise NotImplementedError

    def _expect_obs_term_under_backward(self, obs_term, q_backward):
        return expect_quadratic_term_under_backward(obs_term, q_backward)

    def _expect_obs_term_under_filtering(self, obs_term, q_filtering):
        return expect_quadratic_term_under_filtering(obs_term, q_filtering)

    def _init_V(self, observation, p:GaussianHMM, aux):
        
        quadratic_term = quadratic_term_from_log_gaussian(p.prior)
                                        
        nonlinear_term = self._get_obs_term(observation, p.emission, aux)

        return quadratic_term, nonlinear_term 

    def _update_V(self, observation, quadratic_term, nonlinear_term, q_backward:BackwardParams, p:GaussianHMM, aux):

        dim_z = p.transition.cov.shape[0]

        # integrating all previous terms up to current interation
        integrated_terms = self._integrate_previous_terms(quadratic_term, nonlinear_term, q_backward)

        # adding new transition term already integrated under current backward 
        quadratic_term = integrated_terms + transition_term_integrated_under_backward(q_backward, p.transition)
        
        # integrating the backward under iself only results in constant terms 
        quadratic_term.c += -constant_terms_from_log_gaussian(dim_z, jnp.linalg.det(q_backward.cov)) +  0.5 * dim_z
        
        # adding observation term that will be integrated at next step 
        nonlinear_term = self._get_obs_term(observation, p.emission, aux) 
        
        return quadratic_term, nonlinear_term

    def init(self, first_obs, p:GaussianHMM, q:GaussianHMM, aux):

        quadratic_term, nonlinear_term = self._init_V(first_obs, p, aux)
        q_filtering = init_filtering(first_obs, q.prior, q.emission)

        return quadratic_term, nonlinear_term, q_filtering

    def update(self, observation, quadratic_term, nonlinear_term, q_filtering:FilteringParams, p:GaussianHMM, q:GaussianHMM, aux):
        q_backward = update_backward(q_filtering, q.transition)
        quadratic_term, nonlinear_term = self._update_V(observation, 
                                                quadratic_term, 
                                                nonlinear_term, 
                                                q_backward, 
                                                p,
                                                aux)
        q_filtering = update_filtering(observation, q_filtering, q.transition, q.emission)

        return quadratic_term, nonlinear_term, q_filtering

    def compute(self, observations, p_params, q_params, aux_params=None):

        p = GaussianHMM.build_from_dict(p_params, self.p_def)
        q = GaussianHMM.build_from_dict(q_params, self.q_def)

        aux = self.set_aux(aux_params)

        quadratic_term, nonlinear_term, q_filtering = self.init(observations[0], p, q, aux)
        observations = observations[1:]

        
        # keeping it at hand as an alternative to scan for more verbose debugging
        # for observation in observations:
        #     quadratic_term, nonlinear_term, q_filtering = self.update(observation, 
        #                                                             quadratic_term, 
        #                                                             nonlinear_term, 
        #                                                             q_filtering, 
        #                                                             p, 
        #                                                             q, 
        #                                                             aux)

        def step(carry, x):

            observation = x 
            quadratic_term, nonlinear_term, q_filtering, p, q, aux = carry 

            quadratic_term, nonlinear_term, q_filtering = self.update(observation, 
                                                                    quadratic_term, 
                                                                    nonlinear_term, 
                                                                    q_filtering, 
                                                                    p, 
                                                                    q,
                                                                    aux)

            return (quadratic_term, nonlinear_term, q_filtering, p, q, aux), None

        (quadratic_term, nonlinear_term, q_filtering, p, q, aux), _ = lax.scan(f=step, 
                                    init=(quadratic_term, nonlinear_term, q_filtering, p, q, aux),
                                    xs=observations)

        return self._expect_V_under_filtering(quadratic_term, nonlinear_term, q_filtering)
    
    def _integrate_previous_terms(self, quadratic_term:Collection[QuadForm], nonlinear_term, q_backward:BackwardParams):
        
        result = expect_quadratic_term_under_backward(quadratic_term, q_backward) \
            + self._expect_obs_term_under_backward(nonlinear_term, q_backward)

        return result

    def _expect_V_under_filtering(self, quadratic_term, nonlinear_term, q_filtering:FilteringParams):

        # integrating all previous terms + the nonlinear term that is not integrated yet
        result = expect_quadratic_term_under_filtering(quadratic_term, q_filtering) \
                + self._expect_obs_term_under_filtering(nonlinear_term, q_filtering) \
                - constant_terms_from_log_gaussian(q_filtering.cov.shape[0], jnp.linalg.det(q_filtering.cov)) \
                + 0.5*q_filtering.cov.shape[0]

        return result

class LinearELBO(BackwardELBO):


    def __init__(self, p_def, q_def):
        super().__init__(p_def, q_def)
        
    def _get_obs_term(self, observation, p_emission:GaussianKernel, aux):
        A = p_emission.weight
        b = p_emission.bias - observation
        Omega = p_emission.prec

        result = -0.5*QuadForm.from_A_b_Omega(A, b, Omega)

        result.c += constant_terms_from_log_gaussian(p_emission.cov.shape[0], p_emission.det_cov)

        return result 

class NonLinearELBO(BackwardELBO):

    def __init__(self, p_def, q_def, aux_defs):
        super().__init__(p_def, q_def)
        self.aux_defs = aux_defs

    def set_aux(self, aux_params):
        return {name: Partial(self.aux_defs[name], params=params) for name, params in aux_params.items()}

    def _get_obs_term(self, observation, p_emission, aux):
        v, W =  aux['rec_net'](x=observation)
        return QuadForm(W=W, v=v, c=0.)

def get_elbo(p_def, q_def, aux_defs=None):

    if p_def['transition']['mapping_type'] == 'linear':
        if p_def['emission']['mapping_type'] == 'linear':
            return LinearELBO(p_def, q_def).compute
        elif p_def['emission']['mapping_type'] == 'nonlinear': 
            if aux_defs is None: raise NotImplementedError
            return NonLinearELBO(p_def, q_def, aux_defs).compute
    else: 
        raise NotImplementedError
 