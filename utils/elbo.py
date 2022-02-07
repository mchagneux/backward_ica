from collections import namedtuple
from typing import NamedTuple
import utils.kalman as kalman
from utils.misc import *
import jax.numpy as jnp
from jax.numpy import trace
from jax.numpy.linalg import det, inv

Backward = namedtuple('Backward',['A','a','cov'])
Transition = namedtuple('Transition',['matrix','offset','cov','prec','det_cov'])
Emission = namedtuple('Emission',['matrix','offset','cov','prec','det_cov'])


def _constant_terms_from_log_gaussian(dim, det_cov):
        return -0.5*(dim * jnp.log(2*jnp.pi) + jnp.log(det_cov))
    
def _eval_quad_form(quad_form, x):
    common_term = quad_form.A @ x + quad_form.b
    return common_term.T @ quad_form.Omega @ common_term

def _expect_quad_form_under_backward(quad_form:QuadForm, backward):
    # expectation of (Au+b)^T Omega (Au+b) under the backward 

    constants = trace(quad_form.Omega @ quad_form.A @ backward.cov @ quad_form.A.T)
    quad_form_in_z = QuadForm(Omega=quad_form.Omega, 
                            A=quad_form.A @ backward.A, 
                            b=quad_form.A @ backward.a + quad_form.b)
    return constants, quad_form_in_z

def _expect_transition_quad_form_under_backward(backward, transition):
    # expectation of the quadratic form that appears in the log of the state transition density

    constants =  - 0.5 * trace(transition.prec @ transition.matrix @ backward.cov @ transition.matrix.T)
    quad_form_in_z = QuadForm(Omega=-0.5*transition.prec, 
                            A=transition.matrix @ backward.A - jnp.eye(transition.matrix.shape[0]))
    return constants, quad_form_in_z

def _expect_quad_form_under_filtering(quad_form:QuadForm, filtering):
    return trace(quad_form.Omega @ quad_form.A @ filtering.cov @ quad_form.A) + _eval_quad_form(quad_form, filtering.mean)

def _update_backward(filtering, v_transition):
    
    filtering_prec = inv(filtering.cov)

    backward_prec = v_transition.A.T @ v_transition.prec @ v_transition.A + filtering_prec

    backward_cov = inv(backward_prec)

    common_term = v_transition.A.T @ v_transition.prec 
    A_backward = backward_cov @ common_term
    a_backward = backward_cov @ filtering_prec @ filtering.mean - common_term @  v_transition.a


    return A_backward, a_backward, backward_cov

def _get_quad_form_in_z_obs_term(observation, emission):
    return QuadForm(Omega=-0.5*emission.inv_R, 
                    A = emission.matrix, 
                    b = emission.matrix - observation)    

def _expectations_under_backward(quad_forms, dims, backward, transition, emission, observation):

    new_constants = 0
    new_quad_forms = []

    # dealing with all non-constant terms from previous V: one step before these quad forms were in z_next so they now are quad forms in z that need to be 
    # integrated against the backward, resulting in new quad forms in z_next
    for quad_form in quad_forms: 
        constant, integrated_quad_form = _expect_quad_form_under_backward(quad_form, backward)
        new_constants += constant
        new_quad_forms.append(integrated_quad_form)


    # dealing with observation term seen as a quadratic form in z
    new_constants += _constant_terms_from_log_gaussian(dims.x, emission.det_cov)                                      
    new_quad_forms.append(_get_quad_form_in_z_obs_term(observation, emission))

    # dealing with true transition term whose integration in z_previous under the backward is a quadratic form in z
    new_constants += _constant_terms_from_log_gaussian(dims.z, transition.det_cov)
    constant, integrated_quad_form = _expect_transition_quad_form_under_backward(backward, transition)
    new_constants += constant 
    new_quad_forms.append(integrated_quad_form)

    # dealing with backward term (integration of the quadratic form is just the dimension of z)
    new_constants += -_constant_terms_from_log_gaussian(dims.z, det(backward.cov))
    new_constants += 0.5*dims.z

    
    return new_constants, new_quad_forms


def compute(model:Model, v_model:Model, observations):

    prior = model.prior
    transition = Transition(matrix=model.transition.matrix, offset=model.transition.offset, cov=model.transition.cov, prec=inv(model.transition.cov), det_cov=det(model.transition.cov))
    emission = Emission(matrix=model.observation.matrix, offset=model.observation.offset, cov=model.observation.cov, prec = inv(model.observation.cov), det_cov=inv(model.observation.cov))
    
    v_prior = v_model.prior
    v_transition = Transition(matrix=v_model.transition.matrix, offset= v_model.transition.offset, cov=v_model.transition.cov, prec=inv(v_model.transition.cov), det_cov=det(v_model.transition.cov))
    v_emission = Emission(matrix=v_model.observation.matrix, offset=v_model.observation.offset, cov=v_model.observation.cov, inv_R = inv(v_model.observation.cov), det_cov=inv(v_model.observation.cov))
    
    dims = Dims(z=transition.matrix.shape[0],x=emission.matrix.shape[0])
    filtering = kalman.init(observations[0], v_prior, v_emission)[2:]

    constants = 0
    quad_forms = []

    constants += _constant_terms_from_log_gaussian(dims.z, det(prior.cov)) + \
                 _constant_terms_from_log_gaussian(dims.x, emission.det_cov)
    

    quad_forms.append(_get_quad_form_in_z_obs_term(observations[0], emission))
    quad_forms.append(QuadForm(Omega=-0.5*inv(prior.cov), b=-prior.mean))

    for observation in observations[1:]:

        backward = _update_backward(filtering, v_transition)

        new_constants, new_quad_forms = _expectations_under_backward(quad_forms, 
                                                                        dims,
                                                                        backward,
                                                                        transition,
                                                                        emission, 
                                                                        observation)

        constants += new_constants
        quad_forms = new_quad_forms

        filtering = kalman.filter_step(filtering.mean, 
                                        filtering.cov, 
                                        observation,
                                        v_transition,
                                        v_emission)[2:]

    constants += -_constant_terms_from_log_gaussian(dims.z, det(filtering.cov))

    for quad_form in quad_forms:
        constants += _expect_quad_form_under_filtering(quad_form, filtering)
    
    
    return constants 

            





        



        
        


