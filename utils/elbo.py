from collections import namedtuple
from typing import NamedTuple
import utils.kalman as kalman
from utils.misc import ModelParams, QuadForm, Dims
import jax.numpy as jnp
from jax.numpy import dot, transpose, trace
from jax.numpy.linalg import det, inv

BackwardParams = namedtuple('backward',['A','a','cov'])
Emission = namedtuple('emission',['B','b','R','inv_R','det_R'])
Transition = namedtuple('transition',['A','a','Q','inv_Q','det_Q'])


def _constant_terms_from_log_gaussian(dim, det_cov):
        return -0.5*(dim * jnp.log(2*jnp.pi) + jnp.log(det_cov))
    
def _eval_quad_form(quad_form, x):
    common_term = dot(quad_form.A, x) + quad_form.b
    return dot(transpose(common_term),dot(quad_form.Omega, common_term))

def _expect_quad_form_under_backward(quad_form:QuadForm, backward):
    # expectation of (Au+b)^T Omega (Au+b) under the backward 

    constants = trace(dot(quad_form.Omega, dot(quad_form.A, dot(backward.cov, transpose(quad_form.A)))))
    quad_form_in_z = QuadForm(Omega=quad_form.Omega, 
                            A=dot(quad_form.A, backward.A), 
                            b=dot(quad_form.A, backward.a) + quad_form.b)
    return constants, quad_form_in_z

def _expect_transition_quad_form_under_backward(backward, transition):
    # expectation of the quadratic form that appears in the log of the state transition density

    constants =  - 0.5 * trace(dot(transition.inv_Q, dot(transition.A, dot(backward.cov, transpose(transition.A)))))
    quad_form_in_z = QuadForm(Omega=-0.5*transition.inv_Q, 
                            A=dot(transition.A, backward.A) - jnp.eye(transition.A.shape[0]))
    return constants, quad_form_in_z

def _expect_quad_form_under_filtering(quad_form:QuadForm, filtering):
    return trace(dot(quad_form.Omega, dot(quad_form.A, dot(filtering.cov, transpose(quad_form.A))))) + _eval_quad_form(quad_form, filtering.mean)

def _update_backward(filtering, v_transition):
    
    filtering_prec = inv(filtering.cov)

    backward_prec = dot(transpose(v_transition.A), dot(v_transition.inv_Q, v_transition.A)) + filtering_prec

    backward_cov = inv(backward_prec)

    common_term = dot(transpose(v_transition.A, v_transition.inv_Q)) 
    A_backward = dot(backward_cov, common_term)
    a_backward = dot(backward_cov, dot(filtering_prec, filtering.mean) - dot(common_term, v_transition.a))


    return A_backward, a_backward, backward_cov

def _get_quad_form_in_z_obs_term(observation, emission):
    return QuadForm(Omega=-0.5*emission.inv_R, 
                    A = emission.B, 
                    b = emission.b - observation)    

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
    new_constants += _constant_terms_from_log_gaussian(dims.x, det(emission.R))                                      
    new_quad_forms.append(_get_quad_form_in_z_obs_term(observation, emission))

    # dealing with true transition term whose integration in z_previous under the backward is a quadratic form in z
    new_constants += _constant_terms_from_log_gaussian(dims.z, det(transition.Q))
    constant, integrated_quad_form = _expect_transition_quad_form_under_backward(backward, transition)
    new_constants += constant 
    new_quad_forms.append(integrated_quad_form)

    # dealing with backward term (integration of the quadratic form is just the dimension of z)
    new_constants += -_constant_terms_from_log_gaussian(dims.z, det(backward.cov))
    new_constants += 0.5*dims.z

    
    return new_constants, new_quad_forms


def compute(model:ModelParams, v_model:ModelParams, observations):

    prior = model.prior
    transition = Transition(A=model.transition.matrix, a= model.transition.offset, Q=model.transition.cov, inv_Q=inv(model.transition.cov), det_Q=det(model.transition.cov))
    emission = Emission(B=model.observation.matrix, b=model.observation.offset, R=model.observation.cov, inv_R = inv(model.observation.cov), det_R=inv(model.observation.cov))
    
    v_prior = v_model.prior
    v_transition = Transition(A=v_model.transition.matrix, a= v_model.transition.offset, Q=v_model.transition.cov, inv_Q=inv(v_model.transition.cov), det_Q=det(v_model.transition.cov))
    v_emission = Emission(B=v_model.observation.matrix, b=v_model.observation.offset, R=v_model.observation.cov, inv_R = inv(v_model.observation.cov), det_R=inv(v_model.observation.cov))
    
    dims = Dims(z=transition.A.shape[0],x=emission.B.shape[0])
    filtering = kalman.init(observations[0], v_prior, v_emission)[2:]

    constants = 0
    quad_forms = []

    constants += _constant_terms_from_log_gaussian(dims.z, det(prior.cov)) + \
                 _constant_terms_from_log_gaussian(dims.x, emission.det_R)
    

    quad_forms.append(_get_quad_form_in_z_obs_term(observations[0],emission))
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

            





        



        
        


