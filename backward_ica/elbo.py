from dataclasses import dataclass
from typing import * 
from jax import vmap, lax, config, numpy as jnp
from jax.tree_util import register_pytree_node_class

from backward_ica.hmm import GaussianHMM

from .kalman import filter_step as kalman_filter_step, init as kalman_init
from .utils import *
from typing import * 
config.update("jax_enable_x64", True)


### Some abstractions for frequently used objects when computing elbo via backwards decomposition

@dataclass(init=True, repr=True)
@register_pytree_node_class
class QuadForm:

    A:jnp.ndarray
    b:jnp.ndarray
    Omega:jnp.ndarray 

    def __iter__(self):
        return iter((self.A, self.b, self.Omega))
    
    def tree_flatten(self):
        return ((self.A, self.b, self.Omega), None) 

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def set(self, quad_form, index:int):
        self.A = self.A.at[index].set(quad_form.A)
        self.b = self.b.at[index].set(quad_form.b)
        self.Omega = self.Omega.at[index].set(quad_form.Omega)
        return self

    def get(self, index):
        return QuadForm(self.A[index], self.b[index], self.Omega[index])

    def evaluate(self, x):
        common_term = self.A @ x + self.b
        return common_term.T @ self.Omega @ common_term
        
@dataclass(init=True, repr=True)
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

@dataclass(init=True, repr=True)
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


def _constant_terms_from_log_gaussian(dim:int, det_cov:float)->float:
    """Utility function to compute the log of the term that is against the exponential for a multivariate Normal

    Args:
        dim (int): the dimension of the support of the multivariate Normal
        det_cov (float): the precomputed determinant of the covariance matrix 

    Returns:
        float: the value of the requested factor  
    """

    return -0.5*(dim * jnp.log(2*jnp.pi) + jnp.log(det_cov))

def _expect_obs_term_under_backward(obs_term, q_backward:BackwardParams):
    return _expect_quad_form_under_backward(obs_term, q_backward)

def _expect_obs_term_under_filtering(obs_term, q_filtering:FilteringParams):
    return _expect_quad_form_under_filtering(obs_term, q_filtering)

def _get_obs_term(observation, p_emission:GaussianKernel):
    
    return QuadForm(A=p_emission.weight,
                    b=p_emission.bias - observation,
                    Omega=- 0.5*p_emission.prec)

def _init_filtering(observation, q_prior:Gaussian, q_emission:GaussianKernel):
    return FilteringParams(*kalman_init(observation, q_prior, q_emission)[2:])

def _update_filtering(observation, q_filtering:FilteringParams, q_transition:GaussianKernel, q_emission:GaussianKernel):
    return FilteringParams(*kalman_filter_step(*q_filtering, observation, q_transition, q_emission)[2:])

def _update_backward(q_filtering:FilteringParams, q_transition:GaussianKernel):

    filtering_prec = jnp.linalg.inv(q_filtering.cov)

    backward_prec = q_transition.weight.T @ q_transition.prec @ q_transition.weight + filtering_prec

    cov = jnp.linalg.inv(backward_prec)

    common_term = q_transition.weight.T @ q_transition.prec
    A = cov @ common_term
    a = cov @ (filtering_prec @ q_filtering.mean - common_term @  q_transition.bias)

    return BackwardParams(A,a,cov)

def _integrate_previous_terms(integrate_up_to:int, quadratic_terms:Collection[QuadForm], nonlinear_term, q_backward:BackwardParams):

    quad_forms_transition, quad_forms_emission = quadratic_terms

    masks = jnp.arange(start=0, stop=quad_forms_transition.A.shape[0]) <= integrate_up_to
    _expect_quad_forms_under_backward_masked = vmap(_expect_quad_form_under_backward_masked, in_axes=(0,0,None))
    
    constants0, quad_forms_transition = _expect_quad_forms_under_backward_masked(masks, quad_forms_transition, q_backward)

    masks = masks.at[integrate_up_to].set(False)
    constants1, quad_forms_emission = _expect_quad_forms_under_backward_masked(masks, quad_forms_emission, q_backward)

    constants2, quad_form = _expect_obs_term_under_backward(nonlinear_term, q_backward)
    quad_forms_emission.set(quad_form, index=integrate_up_to)

    return jnp.sum(constants0) + jnp.sum(constants1) + constants2, quad_forms_transition, quad_forms_emission


def _init_V(observation, p:GaussianHMM):

    constants_V = _constant_terms_from_log_gaussian(p.transition.cov.shape[0], jnp.linalg.det(p.prior.cov)) + \
                  _constant_terms_from_log_gaussian(p.emission.cov.shape[0], p.emission.det_cov)
    
    init_transition_term = QuadForm(A=jnp.eye(p.transition.cov.shape[0]),
                                    b=-p.prior.mean, 
                                    Omega=-0.5 * jnp.linalg.inv(p.prior.cov))
    nonlinear_term = _get_obs_term(observation, p.emission)

    return constants_V, init_transition_term, nonlinear_term 

def _expect_transition_quad_form_under_backward(q_backward:BackwardParams, p_transition:GaussianKernel):
    # expectation of the quadratic form that appears in the log of the state transition density

    A=p_transition.weight @ q_backward.A - jnp.eye(p_transition.cov.shape[0])
    b=p_transition.weight @ q_backward.a + p_transition.bias
    Omega = -0.5*p_transition.prec

    return QuadForm(A,b,Omega)

def _no_integration(quad_form, q_backward):
    return 0.0, quad_form

def _expect_quad_form_under_backward(quad_form:QuadForm, q_backward:BackwardParams):
    # the result is still a quadratic forms with new parameters, following the formula for expected values of quadratic forms  
    Sigma = quad_form.A @ q_backward.cov @ quad_form.A.T
    constant = jnp.trace(quad_form.Omega @ Sigma)
    A = quad_form.A @ q_backward.A
    b = quad_form.A @ q_backward.a + quad_form.b
    Omega = quad_form.Omega
    return constant, QuadForm(A,b,Omega)

def _expect_quad_form_under_backward_masked(mask, quad_form:QuadForm, q_backward:BackwardParams):
    # if mask: 
    #     return _expect_quad_form_under_backward(quad_form, q_backward)

    # return _no_integration(quad_form, q_backward)
    return lax.cond(mask, _expect_quad_form_under_backward, _no_integration, quad_form, q_backward)

def _update_V(observation, integrate_up_to, quadratic_terms, nonlinear_term, q_backward:BackwardParams, p:GaussianHMM):

    dim_z, dim_x = p.transition.cov.shape[0], p.emission.cov.shape[0]

    # integrating all previous terms up to current interation
    constants, quad_forms_transition, quad_forms_emission =  _integrate_previous_terms(integrate_up_to, quadratic_terms, nonlinear_term, q_backward)

    # the backward term integrated under itself integrates to constant terms 
    constants +=  - _constant_terms_from_log_gaussian(dim_z, jnp.linalg.det(q_backward.cov)) \
                +  0.5 * dim_z - 0.5 * jnp.trace(p.transition.prec @ p.transition.weight @ q_backward.cov @ p.transition.weight.T) 



    # integrating new transition term under new backward
    constants += _constant_terms_from_log_gaussian(dim_z, p.transition.det_cov) 
    new_quad_form_transition = _expect_transition_quad_form_under_backward(q_backward, p.transition)
    quad_forms_transition.set(new_quad_form_transition, index=integrate_up_to+1) 

    # adding observation term that will be integrated at next step 
    constants += _constant_terms_from_log_gaussian(dim_x, p.emission.det_cov)
    nonlinear_term = _get_obs_term(observation, p.emission) 

    return constants, [quad_forms_transition, quad_forms_emission], nonlinear_term

def _expect_quad_form_under_filtering(quad_form:QuadForm, q_filtering:FilteringParams):
    return jnp.trace(quad_form.Omega @ quad_form.A @ q_filtering.cov @ quad_form.A.T) + quad_form.evaluate(q_filtering.mean)

def _expect_V_under_filtering(constants_V, quadratic_terms, nonlinear_term, q_filtering:FilteringParams):
    result = constants_V

    quad_forms_transition, quad_forms_emission = quadratic_terms

    # the last term in the emission terms is still adummy term because it has not been integrated yet, we remove it 
    quad_forms_emission.A = quad_forms_emission.A[:-1]
    quad_forms_emission.b = quad_forms_emission.b[:-1]
    quad_forms_emission.Omega = quad_forms_emission.Omega[:-1]

    expect_quad_forms_under_filtering = vmap(_expect_quad_form_under_filtering, in_axes=(0, None))

    # integrating all previous terms + the nonlinear term that is not integrated yet
    result += jnp.sum(expect_quad_forms_under_filtering(quad_forms_transition, q_filtering)) \
            + jnp.sum(expect_quad_forms_under_filtering(quad_forms_emission, q_filtering)) \
            + _expect_obs_term_under_filtering(nonlinear_term, q_filtering)

    # the last term that appears in the telescopating series
    result += 0.5*q_filtering.mean.shape[0]

    return result

def init(observations, p:GaussianHMM, q:GaussianHMM):

    constants_V, init_transition_term, nonlinear_term = _init_V(observations[0], p)
    q_filtering = _init_filtering(observations[0], q.prior, q.emission)
    dim_z = p.transition.cov.shape[0]

    num_terms = len(observations)
    quad_forms_transition = QuadForm(A=jnp.empty(shape=(num_terms, dim_z, dim_z)), 
                                    b=jnp.empty(shape=(num_terms, dim_z)),
                                    Omega=jnp.empty(shape=(num_terms, dim_z, dim_z))).set(init_transition_term,index=0)

    dim_x = p.emission.cov.shape[0]

    quad_forms_emission = QuadForm(A=jnp.empty(shape=(num_terms, dim_x, dim_z)),
                                b=jnp.empty(shape=(num_terms, dim_x)),
                                Omega=jnp.empty(shape=(num_terms, dim_x, dim_x)))
            
    quadratic_terms = [quad_forms_transition, quad_forms_emission]
    return constants_V, quadratic_terms, nonlinear_term, q_filtering

def _update(observation, integrate_up_to, quadratic_terms, nonlinear_term, q_filtering:FilteringParams, p:GaussianHMM, q:GaussianHMM):
    q_backward = _update_backward(q_filtering, q.transition)
    constants, quadratic_terms, nonlinear_term = _update_V(observation, 
                                                    integrate_up_to,
                                                    quadratic_terms, 
                                                    nonlinear_term, 
                                                    q_backward, 
                                                    p)

    q_filtering = _update_filtering(observation, q_filtering, q.transition, q.emission)

    return constants, quadratic_terms, nonlinear_term, q_filtering

def linear_gaussian_elbo(observations, p_raw, q_raw, p_aux_info, q_aux_info):

    p = GaussianHMM.build_from_dict(p_raw, p_aux_info)
    q = GaussianHMM.build_from_dict(q_raw, q_aux_info) 
    
    constants_V, quadratic_terms, nonlinear_term, q_filtering = init(observations, p, q)

    observations = observations[1:]

    integrate_up_to_array = jnp.arange(start=0, stop=len(observations))

    
    ## keeping it at hand as an alternative to scan for more verbose debugging
    # for observation, integrate_up_to in zip(observations, integrate_up_to_array):
    #     new_constants, quadratic_terms, nonlinear_term, q_filtering = _update(observation, integrate_up_to, quadratic_terms, nonlinear_term, q_filtering, p, q)
    #     constants_V += new_constants

    def V_step(carry, x):

        observation, integrate_up_to = x 
        quadratic_terms, nonlinear_term, q_filtering, p, q = carry 

        new_constants, quadratic_terms, nonlinear_term, q_filtering = _update(observation, 
                                                                    integrate_up_to,
                                                                    quadratic_terms, 
                                                                    nonlinear_term, 
                                                                    q_filtering, 
                                                                    p, 
                                                                    q)

        return (quadratic_terms, nonlinear_term, q_filtering, p, q), new_constants

    (quadratic_terms, nonlinear_term, q_filtering, p, q), constants = lax.scan(f=V_step, 
                                init=(quadratic_terms, nonlinear_term, q_filtering, p, q),
                                xs=(observations, integrate_up_to_array))

    constants_V += jnp.sum(constants) 
    constants_V += -_constant_terms_from_log_gaussian(p.transition.cov.shape[0], jnp.linalg.det(q_filtering.cov))

    return _expect_V_under_filtering(constants_V, quadratic_terms, nonlinear_term, q_filtering)
    

    


        
