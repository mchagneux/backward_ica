from dataclasses import dataclass
from turtle import back, backward
from typing import * 
from jax import vmap, lax, config, numpy as jnp
from jax.tree_util import register_pytree_node_class, Partial
from .hmm import GaussianHMM

from .kalman import filter_step as kalman_filter_step, init as kalman_init
from .utils import GaussianKernel, _mappings, LinearGaussianKernel, Gaussian, prec_and_det, LinearGaussianKernel
from typing import * 
from abc import ABCMeta, abstractmethod
from functools import partial
config.update("jax_enable_x64", True)
import numpy as np 
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

class ELBO(metaclass=ABCMeta): 

    def __init__(self, p_def, q_def):
        self.p_def = p_def 
        self.q_def = q_def 

    @abstractmethod
    def _get_emission_term(self, *args):
        raise NotImplementedError 

    @abstractmethod
    def compute(self, observations, p_params, q_params, rec_net_params=None):
        raise NotImplementedError
    
    def _expect_emission_term_under_backward(self, emission_term, q_backward):
        return expect_quadratic_term_under_backward(emission_term, q_backward)

    def _expect_emission_term_under_filtering(self, emission_term, q_filtering):
        return expect_quadratic_term_under_filtering(emission_term, q_filtering)

    def _integrate_previous_terms(self, quadratic_term:Collection[QuadTerm], nonlinear_term, q_backward:LinearGaussianKernel):
        
        result = expect_quadratic_term_under_backward(quadratic_term, q_backward) \
            + self._expect_emission_term_under_backward(nonlinear_term, q_backward)

        return result

    def _expect_V_under_filtering(self, quadratic_term, nonlinear_term, q_filtering:Gaussian):

        # integrating all previous terms + the nonlinear term that is not integrated yet
        result = expect_quadratic_term_under_filtering(quadratic_term, q_filtering) \
                + self._expect_emission_term_under_filtering(nonlinear_term, q_filtering) \
                - constant_terms_from_log_gaussian(q_filtering.cov.shape[0], q_filtering.det_cov) \
                + 0.5*q_filtering.cov.shape[0]

        return result
        
    def _init_V(self, observation, p:GaussianHMM, rec_net_params):
        
        quadratic_term = quadratic_term_from_log_gaussian(p.prior)
                                        
        nonlinear_term = self._get_emission_term(observation, p.emission, rec_net_params)

        return quadratic_term, nonlinear_term 

    def _update_V(self, observation, quadratic_term, nonlinear_term, q_backward:LinearGaussianKernel, p:GaussianHMM, rec_net_params):

        dim_z = p.transition.cov.shape[0]

        # integrating all previous terms up to current interation
        integrated_terms = self._integrate_previous_terms(quadratic_term, nonlinear_term, q_backward)

        # adding new transition term already integrated under current backward 
        quadratic_term = integrated_terms + transition_term_integrated_under_backward(q_backward, p.transition)
        
        # integrating the backward under iself only results in constant terms 
        quadratic_term.c += -constant_terms_from_log_gaussian(dim_z, jnp.linalg.det(q_backward.cov)) +  0.5 * dim_z
        
        # adding observation term that will be integrated at next step 
        nonlinear_term = self._get_emission_term(observation, p.emission, rec_net_params) 
        
        return quadratic_term, nonlinear_term

class LinearELBO(ELBO):


    def __init__(self, p_def, q_def):
        super().__init__(p_def, q_def)
        
    def _get_emission_term(self, observation, p_emission:LinearGaussianKernel, rec_net):
        A = p_emission.weight
        b = p_emission.bias - observation
        Omega = p_emission.prec

        result = -0.5*QuadTerm.from_A_b_Omega(A, b, Omega)

        result.c += constant_terms_from_log_gaussian(p_emission.cov.shape[0], p_emission.det_cov)

        return result 

    def init_filtering(self, observation, q_prior:Gaussian, q_emission:LinearGaussianKernel):
        mean, cov = kalman_init(observation, q_prior, q_emission)[2:]
        return Gaussian(mean, cov, *prec_and_det(cov))

    def update_filtering(self, observation, q_filtering:Gaussian, q_transition:LinearGaussianKernel, q_emission:LinearGaussianKernel):
        mean, cov = kalman_filter_step(q_filtering.mean, q_filtering.cov, observation, q_transition, q_emission)[2:]
        return Gaussian(mean, cov, *prec_and_det(cov))

    def update_backward(self, q_filtering:Gaussian, q_transition:LinearGaussianKernel):

        prec = q_transition.weight.T @ q_transition.prec @ q_transition.weight + q_filtering.prec
        cov = jnp.linalg.inv(prec)

        common_term = q_transition.weight.T @ q_transition.prec
        A = cov @ common_term
        a = cov @ (q_filtering.prec @ q_filtering.mean - common_term @  q_transition.bias)

        return LinearGaussianKernel(mapping=_mappings['linear'], 
                            mapping_params={'weight':A, 'bias':a},
                            cov=cov,
                            prec=prec, 
                            det_cov=jnp.linalg.det(cov))

    def compute(self, observations, p_params, q_params, rec_net_params=None):


        p = GaussianHMM.build_from_dict(p_params, self.p_def)
        q = GaussianHMM.build_from_dict(q_params, self.q_def)

        quadratic_term, nonlinear_term = self._init_V(observations[0], p, rec_net_params)
        q_filtering = self.init_filtering(observations[0], q.prior, q.emission)

        q_filtering_means = [q_filtering.mean.copy()]
        q_filtering_covs = [q_filtering.cov.copy()]

        observations = observations[1:]

        def step(carry, x):

            observation = x 
            quadratic_term, nonlinear_term, q_filtering, p, q, rec_net_params = carry 

            q_backward = self.update_backward(q_filtering, q.transition)
            quadratic_term, nonlinear_term = self._update_V(observation, 
                                                    quadratic_term, 
                                                    nonlinear_term, 
                                                    q_backward, 
                                                    p,
                                                    rec_net_params)
            q_filtering = self.update_filtering(observation, q_filtering, q.transition, q.emission)

            return (quadratic_term, nonlinear_term, q_filtering, p, q, rec_net_params), (q_backward, q_filtering)

        (quadratic_term, nonlinear_term, q_filtering, p, q, rec_net_params), (q_backwards, q_filterings)  = lax.scan(f=step, 
                                    init=(quadratic_term, nonlinear_term, q_filtering, p, q, rec_net_params),
                                    xs=observations)
        
        # q_filtering_means.extend([mean for mean in q_filterings.mean])
        # q_filtering_covs.extend([cov for cov in q_filterings.cov])
        # q_backward_As = [A for A in q_backwards.weight]
        # q_backward_as = [a for a in q_backwards.bias]
        # q_backward_covs = [cov for cov in q_backwards.cov]

        return -self._expect_V_under_filtering(quadratic_term, nonlinear_term, q_filtering)#, (q_filtering_means, q_filtering_covs, q_backward_As, q_backward_as, q_backward_covs)

class LinearELBOWithRecNet(ELBO):


    def __init__(self, p_def, q_def):
        super().__init__(p_def, q_def)
        
    def _get_emission_term(self, observation, p_emission:GaussianKernel, rec_net_params):
        eta1, eta2 = self.rec_net_def(x=observation, params=rec_net_params)
        const = -0.25 * eta1.T @ jnp.linalg.solve(eta2, eta1) - 0.5 * jnp.log(jnp.linalg.det(-2*eta2)) - eta1.shape[0] * jnp.log(jnp.pi)
        result = QuadTerm(W=eta2, 
                        v=eta1, 
                        c=const)
        return result 

    def init_filtering(self, observation, q_prior:Gaussian, q_emission:LinearGaussianKernel):
        mean, cov = kalman_init(observation, q_prior, q_emission)[2:]
        return Gaussian(mean, cov, *prec_and_det(cov))

    def update_filtering(self, observation, q_filtering:Gaussian, q_transition:LinearGaussianKernel, q_emission:LinearGaussianKernel):
        mean, cov = kalman_filter_step(q_filtering.mean, q_filtering.cov, observation, q_transition, q_emission)[2:]
        return Gaussian(mean, cov, *prec_and_det(cov))

    def update_backward(self, q_filtering:Gaussian, q_transition:LinearGaussianKernel):

        prec = q_transition.weight.T @ q_transition.prec @ q_transition.weight + q_filtering.prec
        cov = jnp.linalg.inv(prec)

        common_term = q_transition.weight.T @ q_transition.prec
        A = cov @ common_term
        a = cov @ (q_filtering.prec @ q_filtering.mean - common_term @  q_transition.bias)

        return LinearGaussianKernel(mapping=_mappings['linear'], 
                            mapping_params={'weight':A, 'bias':a},
                            cov=cov,
                            prec=prec, 
                            det_cov=jnp.linalg.det(cov))

    def compute(self, observations, p_params, q_params, rec_net_params):


        p = GaussianHMM.build_from_dict(p_params, self.p_def)
        q = GaussianHMM.build_from_dict(q_params, self.q_def)

        quadratic_term, nonlinear_term = self._init_V(observations[0], p, rec_net_params)
        q_filtering = self.init_filtering(observations[0], q.prior, q.emission)

        
        observations = observations[1:]

        def step(carry, x):

            observation = x 
            quadratic_term, nonlinear_term, q_filtering, p, q, rec_net_params = carry 

            q_backward = self.update_backward(q_filtering, q.transition)
            quadratic_term, nonlinear_term = self._update_V(observation, 
                                                    quadratic_term, 
                                                    nonlinear_term, 
                                                    q_backward, 
                                                    p,
                                                    rec_net_params)
            q_filtering = self.update_filtering(observation, q_filtering, q.transition, q.emission)

            return (quadratic_term, nonlinear_term, q_filtering, p, q, rec_net_params), None

        (quadratic_term, nonlinear_term, q_filtering, p, q, rec_net_params), _  = lax.scan(f=step, 
                                    init=(quadratic_term, nonlinear_term, q_filtering, p, q, rec_net_params),
                                    xs=observations)

        return -self._expect_V_under_filtering(quadratic_term, nonlinear_term, q_filtering)

class NonLinearELBO(ELBO):

    def __init__(self, p_def, q_def, rec_net_def):
        super().__init__(p_def, q_def)
        self.rec_net_def = rec_net_def


    def _get_emission_term(self, observation, p_emission:GaussianKernel, rec_net_params):
        eta1, eta2 = self.rec_net_def(x=observation, params=rec_net_params)
        const = -0.25 * eta1.T @ jnp.linalg.solve(eta2, eta1) - 0.5 * jnp.log(jnp.linalg.det(-2*eta2)) - eta1.shape[0] * jnp.log(jnp.pi)
        result = QuadTerm(W=eta2, 
                        v=eta1, 
                        c=const)
        return result 

    def _expect_emission_term_under_backward(self, emission_term, q_backward):
        return expect_quadratic_term_under_backward(emission_term, q_backward)

    def _expect_emission_term_under_filtering(self, emission_term, q_filtering):
        return expect_quadratic_term_under_filtering(emission_term, q_filtering)

    def init_filtering(self, observation, filtering_init_params):
        
        mean, cov = self.q_def['filtering']['init'](observation=observation, 
                                                params=filtering_init_params)

        return Gaussian(mean, cov, *prec_and_det(cov))

    def update_filtering(self, observation, q_filtering:Gaussian, filtering_update_params):
        mean, cov = self.q_def['filtering']['update'](observation=observation, 
                                                    filtering_mean=q_filtering.mean, 
                                                    filtering_cov=q_filtering.cov, 
                                                    params=filtering_update_params)

        return Gaussian(mean, cov, *prec_and_det(cov))

    def update_backward(self, q_filtering:Gaussian, backward_params):

        A, a, cov = self.q_def['backward'](filtering_mean=q_filtering.mean, 
                                        filtering_cov=q_filtering.cov, 
                                        params=backward_params)

        return LinearGaussianKernel(_mappings['linear'], 
                                    {'weight':A, 'bias':a},
                                    cov,
                                    *prec_and_det(cov))

    def compute(self, observations, p_params, q_params, rec_net_params):

        p = GaussianHMM.build_from_dict(p_params, self.p_def)

        quadratic_term, nonlinear_term = self._init_V(observations[0], p, rec_net_params)
        q_filtering = self.init_filtering(observations[0], q_params['filtering']['init'])

        observations = observations[1:]

        def step(carry, x):

            observation = x 
            quadratic_term, nonlinear_term, q_filtering, p, q_params, rec_net_params = carry 

            q_backward = self.update_backward(q_filtering, q_params['backward'])
            quadratic_term, nonlinear_term = self._update_V(observation, 
                                                    quadratic_term, 
                                                    nonlinear_term, 
                                                    q_backward, 
                                                    p,
                                                    rec_net_params)
            q_filtering = self.update_filtering(observation, q_filtering, q_params['filtering']['update'])

            return (quadratic_term, nonlinear_term, q_filtering, p, q_params, rec_net_params), None

        (quadratic_term, nonlinear_term, q_filtering, p, q_params, rec_net_params), _  = lax.scan(f=step, 
                                    init=(quadratic_term, nonlinear_term, q_filtering, p, q_params, rec_net_params),
                                    xs=observations)

        return -self._expect_V_under_filtering(quadratic_term, nonlinear_term, q_filtering)
    


# def get_neg_elbo(p_def, q_def, aux_defs=None):

#     if p_def['transition']['mapping_type'] == 'linear':
#         if p_def['emission']['mapping_type'] == 'linear':
#             return LinearELBO(p_def, q_def).compute
#         elif p_def['emission']['mapping_type'] == 'nonlinear': 
#             if aux_defs is None: raise NotImplementedError
#             return NonLinearELBO(p_def, q_def, aux_defs).compute
    # else: 
    #     raise NotImplementedError
 