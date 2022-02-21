from collections import namedtuple
from turtle import back, backward
from typing import NamedTuple
import src.kalman as kalman 
from .misc import *
import jax
import jax.numpy as jnp
from jax.numpy import trace
from jax.numpy.linalg import det, inv
from jax import jit
from jax.experimental import loops


Backward = namedtuple('Backward',['A','a','cov'])
Filtering = namedtuple('Filtering',['mean','cov'])

Transition = namedtuple('Transition',['matrix','offset','cov','prec','det_cov'])
Emission = namedtuple('Emission',['matrix','offset','cov','prec','det_cov'])



def _sum_quad_forms(quad_forms):
    transformed_quad_forms = []
    ndim_A = quad_forms[0].A.shape[0]
    for quad_form in quad_forms: 
        transformed_quad_forms.append(QuadForm(Omega=quad_form.A.T @ quad_form.Omega @ quad_form.A, A = jnp.eye(ndim_A), b = jnp.linalg.solve(quad_form.A, quad_form.b)))
    
    Omega_sum = sum(quad_form.Omega for quad_form in transformed_quad_forms)
    b_sum = jnp.linalg.solve(Omega_sum, sum(quad_form.b @ quad_form.Omega for quad_form in transformed_quad_forms))

    return QuadForm(Omega = Omega_sum, A = jnp.eye(ndim_A), b = b_sum)

def _constant_terms_from_log_gaussian(dim, det_cov):
        return -0.5*(dim * jnp.log(2*jnp.pi) + jnp.log(det_cov))
    
def _eval_quad_form(A, b, Omega, x):
    common_term = A @ x + b
    return common_term.T @ Omega @ common_term

def _expect_quad_form_under_backward(A, b, Omega, backward_A, backward_a, backward_cov):
    # expectation of (Au+b)^T Omega (Au+b) under the backward 

    constants = trace(Omega @ A @ backward_cov @ A.T)
    A = A @ backward_A
    b = A @ backward_a + b
    return constants, A, b, Omega


def _expect_transition_quad_form_under_backward(backward_A, backward_a, backward_cov, transition):
    # expectation of the quadratic form that appears in the log of the state transition density

    Omega=-0.5*transition.prec
    A = transition.matrix @ backward_A - jnp.eye(transition.matrix.shape[0])
    b = transition.matrix @ backward_a + transition.offset
    return A, b, Omega


def _expect_quad_form_under_filtering(A, b, Omega, filtering_mean, filtering_cov):

    return trace(Omega @ A @ filtering_cov @ A.T) + _eval_quad_form(A, b, Omega, filtering_mean)


def _update_backward(filtering_mean, filtering_cov, v_transition):
    
    filtering_prec = inv(filtering_cov)

    backward_prec = v_transition.matrix.T @ v_transition.prec @ v_transition.matrix + filtering_prec

    backward_cov = inv(backward_prec)

    common_term = v_transition.matrix.T @ v_transition.prec 
    A_backward = backward_cov @ common_term
    a_backward = backward_cov @ (filtering_prec @ filtering_mean - common_term @  v_transition.offset)


    return A_backward, a_backward, backward_cov

def _get_quad_form_in_z_obs_term(observation, emission):
    Omega=-0.5*emission.prec
    A = emission.matrix
    b = emission.offset - observation
    return A, b, Omega




def _terms_from_single_step(backward_A, backward_a, backward_cov, observation, dims, transition, emission):

    # dealing with true transition term whose integration in z_previous under the backward is a quadratic form in z
    new_constants = _constant_terms_from_log_gaussian(dims.z, transition.det_cov)
    new_transition_term = _expect_transition_quad_form_under_backward(backward_A, backward_a, backward_cov, transition)
    new_constants += - 0.5 * jnp.trace(transition.prec @ transition.matrix @ backward_cov @ transition.prec.T)

    # dealing with observation term seen as a quadratic form in z
    new_constants += _constant_terms_from_log_gaussian(dims.x, emission.det_cov)                                      
    new_obs_term = _get_quad_form_in_z_obs_term(observation, emission)

    # dealing with backward term (integration of the quadratic form is just the dimension of z)
    new_constants += -_constant_terms_from_log_gaussian(dims.z, det(backward_cov))
    new_constants += 0.5*dims.z

    
    return new_constants, *new_transition_term, *new_obs_term

def _no_integration(A, b, Omega, backward_A, backward_a, backward_cov):
    return 0.0, A, b, Omega

def _backward_integration_step(carry, x):
    constants, A, b, Omega = carry
    backward_A, backward_a, backward_cov, flag = x

    new_constants, A, b, Omega = jax.lax.cond(flag, 
                                            _expect_quad_form_under_backward,
                                            _no_integration, 
                                            A, b, Omega, backward_A, backward_a, backward_cov)
                                                
    return (constants+new_constants, A, b, Omega), None

def _integrate_term_under_backward(A, b, Omega, index_for_mask, backward_As, backward_as, backward_covs):

    mask = jnp.arange(0, len(backward_As)) >= index_for_mask 
    return jax.lax.scan(f=_backward_integration_step, init=(0.0, A, b, Omega), xs=(backward_As, backward_as, backward_covs, mask))[0]


def linear_gaussian_elbo(model:Model, v_model:Model, observations):


    prior = model.prior
    transition = Transition(matrix=model.transition.matrix, 
                            offset=model.transition.offset, 
                            cov=model.transition.cov, 
                            prec=inv(model.transition.cov), 
                            det_cov=det(model.transition.cov))

    emission = Emission(matrix=model.emission.matrix, 
                        offset=model.emission.offset, 
                        cov=model.emission.cov, 
                        prec = inv(model.emission.cov), 
                        det_cov=det(model.emission.cov))
    
    v_transition = Transition(matrix=v_model.transition.matrix, 
                            offset=v_model.transition.offset, 
                            cov=v_model.transition.cov, 
                            prec=inv(v_model.transition.cov), 
                            det_cov=det(v_model.transition.cov))

    
    dims = Dims(z=transition.matrix.shape[0], x=emission.matrix.shape[0])


    filtering_means, filtering_covs, _  = kalman.filter(observations, v_model)

    backward_As, backward_as, backward_covs = jax.vmap(_update_backward, in_axes=(0,0,None))(filtering_means[:-1], 
                                                                                            filtering_covs[:-1], 
                                                                                            v_transition)


    constants = _constant_terms_from_log_gaussian(dims.z, det(prior.cov)) + \
                _constant_terms_from_log_gaussian(dims.x, emission.det_cov)
                
    init_transition_term = (jnp.eye(dims.z), -prior.mean, -0.5*inv(prior.cov))
    init_obs_term = _get_quad_form_in_z_obs_term(observations[0], emission)

    new_terms  = jax.vmap(_terms_from_single_step, in_axes=(0,0,0,0,None,None,None))(backward_As, 
                                                                                    backward_as, 
                                                                                    backward_covs, 
                                                                                    observations[1:], 
                                                                                    dims, transition, emission)

    constants += jnp.sum(new_terms[0])
    transition_terms = new_terms[1:4]
    obs_terms = new_terms[4:]


    transition_terms = (jnp.concatenate((jnp.expand_dims(init_transition_term[0], axis=0), transition_terms[0])),
                        jnp.concatenate((jnp.expand_dims(init_transition_term[1], axis=0), transition_terms[1])),
                        jnp.concatenate((jnp.expand_dims(init_transition_term[2], axis=0), transition_terms[2])))


    obs_terms = (jnp.concatenate((jnp.expand_dims(init_obs_term[0], axis=0), obs_terms[0])),
                jnp.concatenate((jnp.expand_dims(init_obs_term[1], axis=0), obs_terms[1])),
                jnp.concatenate((jnp.expand_dims(init_obs_term[2], axis=0), obs_terms[2])))


    
    num_backwards = len(backward_As)

    _integrate_all_terms = jax.vmap(_integrate_term_under_backward, in_axes=(0,0,0,0, None, None, None))
    _integrate_against_filtering = jax.vmap(_expect_quad_form_under_filtering, in_axes=(0, 0, 0, None, None))

    indices_for_masks = jnp.arange(0, num_backwards)

    constants_from_integration, As, bs, Omegas = _integrate_all_terms(transition_terms[0][:-1], transition_terms[1][:-1], transition_terms[2][:-1], indices_for_masks, backward_As, backward_as, backward_covs)
    constants += jnp.sum(constants_from_integration)
    constants += jnp.sum(_integrate_against_filtering(As, bs, Omegas, filtering_means[-1], filtering_covs[-1]))
    constants += _expect_quad_form_under_filtering(transition_terms[0][-1], transition_terms[1][-1], transition_terms[2][-1], filtering_means[-1], filtering_covs[-1])

    constants_from_integration, As, bs, Omegas = _integrate_all_terms(obs_terms[0][:-1], obs_terms[1][:-1], obs_terms[2][:-1], indices_for_masks, backward_As, backward_as, backward_covs)
    constants += jnp.sum(constants_from_integration)
    constants += jnp.sum(_integrate_against_filtering(As, bs, Omegas, filtering_means[-1], filtering_covs[-1]))
    constants += _expect_quad_form_under_filtering(obs_terms[0][-1], obs_terms[1][-1], obs_terms[2][-1], filtering_means[-1], filtering_covs[-1])

    constants += -_constant_terms_from_log_gaussian(dims.z, det(filtering_covs[-1])) + 0.5*dims.z
    
    return constants








        



        
        


