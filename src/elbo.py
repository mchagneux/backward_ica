from abc import abstractmethod, abstractstaticmethod
import src.kalman as kalman
from src.misc import Model, QuadForm, actual_model_from_raw_parameters
import jax.numpy as jnp
from collections import namedtuple
import jax 
from jax import lax 
jax.config.update("jax_enable_x64", True)

Backward = namedtuple('Backward', ['A','a','cov'])
Filtering = namedtuple('Filtering', ['mean','cov'])
Transition = namedtuple('Transition',['weight', 'bias', 'cov', 'prec', 'det_cov'])
Emission = namedtuple('Emission',['weight', 'bias', 'cov', 'prec', 'det_cov'])


def _constant_terms_from_log_gaussian(dim, det_cov):
    return -0.5*(dim * jnp.log(2*jnp.pi) + jnp.log(det_cov))

def _eval_quad_form(quad_form, x):
    common_term = quad_form.A @ x + quad_form.b
    return common_term.T @ quad_form.Omega @ common_term

def _expect_obs_term_under_backward(obs_term, q_backward):
    return _expect_quad_form_under_backward(obs_term, q_backward)

def _expect_obs_term_under_filtering(obs_term, q_filtering):
    return _expect_quad_form_under_filtering(obs_term, q_filtering)

def _get_obs_term(observation, p_emission):
    A = p_emission.weight
    b = p_emission.bias - observation
    Omega= - 0.5*p_emission.prec
    return QuadForm(A=A, b=b, Omega=Omega)

def _init_filtering(observation, q_prior, q_emission):
    return Filtering(*kalman.init(observation, q_prior, q_emission)[2:])

def _update_filtering(observation, q_filtering, q_transition, q_emission):
    return Filtering(*kalman.filter_step(*q_filtering, observation, q_transition, q_emission)[2:])

def _update_backward(q_filtering, q_transition):

    filtering_prec = jnp.linalg.inv(q_filtering.cov)

    backward_prec = q_transition.weight.T @ q_transition.prec @ q_transition.weight + filtering_prec

    cov = jnp.linalg.inv(backward_prec)

    common_term = q_transition.weight.T @ q_transition.prec 
    A = cov @ common_term
    a = cov @ (filtering_prec @ q_filtering.mean - common_term @  q_transition.bias)

    return Backward(A=A, a=a, cov=cov)

def _integrate_previous_terms(integrate_up_to, quad_forms, nonlinear_term, q_backward):

    masks = jnp.arange(start=0, stop=quad_forms.A.shape[0]) <= integrate_up_to
    constants0, quad_forms = jax.vmap(_expect_quad_form_under_backward_masked, in_axes=(0,0,None))(masks, quad_forms, q_backward)

    constants1, quad_form = _expect_obs_term_under_backward(nonlinear_term, q_backward)
    
    As = quad_forms.A.at[integrate_up_to+1].set(quad_form.A)
    bs = quad_forms.b.at[integrate_up_to+1].set(quad_form.b)
    Omegas = quad_forms.Omega.at[integrate_up_to+1].set(quad_form.Omega)

    return jnp.sum(constants0) + constants1, QuadForm(A=As, b=bs, Omega=Omegas)


def _init_V(observation, p):

    constants_V = _constant_terms_from_log_gaussian(p.transition.cov.shape[0], jnp.linalg.det(p.prior.cov)) + \
                  _constant_terms_from_log_gaussian(p.emission.cov.shape[0], p.emission.det_cov)
    
    A, b, Omega = jnp.eye(p.transition.cov.shape[0]), -p.prior.mean, -0.5 * jnp.linalg.inv(p.prior.cov)
    nonlinear_term = _get_obs_term(observation, p.emission)

    return constants_V, A, b, Omega, nonlinear_term 

def _expect_transition_quad_form_under_backward(q_backward, p_transition):
    # expectation of the quadratic form that appears in the log of the state transition density

    A=p_transition.weight @ q_backward.A - jnp.eye(p_transition.cov.shape[0])
    b=p_transition.weight @ q_backward.a + p_transition.bias
    Omega = -0.5*p_transition.prec

    return QuadForm(A=A, b=b, Omega=Omega) 

def _no_integration(quad_form, q_backward):
    return 0.0, quad_form

def _expect_quad_form_under_backward(quad_form, q_backward):
    constant = jnp.trace(quad_form.Omega @ quad_form.A @ q_backward.cov @ quad_form.A.T)
    A = quad_form.A @ q_backward.A
    b = quad_form.A @ q_backward.a + quad_form.b
    Omega = quad_form.Omega
    return constant, QuadForm(A=A, b=b, Omega=Omega)

def _expect_quad_form_under_backward_masked(mask, quad_form, q_backward):
    return lax.cond(mask, _expect_quad_form_under_backward, _no_integration, quad_form, q_backward)

def _update_V(observation, integrate_up_to, quad_forms, nonlinear_term, q_backward, p):

    constants, quad_forms =  _integrate_previous_terms(integrate_up_to, quad_forms, nonlinear_term, q_backward)
    dim_z, dim_x = p.transition.cov.shape[0], p.emission.cov.shape[0]
    constants +=  _constant_terms_from_log_gaussian(dim_x, p.emission.det_cov) \
                +   _constant_terms_from_log_gaussian(dim_z, p.transition.det_cov) \
                + - _constant_terms_from_log_gaussian(dim_z, jnp.linalg.det(q_backward.cov)) \
                +  0.5 * dim_z \
                + - 0.5 * jnp.trace(p.transition.prec @ p.transition.weight @ q_backward.cov @ p.transition.weight.T)

    quad_form = _expect_transition_quad_form_under_backward(q_backward, p.transition)
    
    As = quad_forms.A.at[integrate_up_to+2].set(quad_form.A)
    bs = quad_forms.b.at[integrate_up_to+2].set(quad_form.b)
    Omegas = quad_forms.Omega.at[integrate_up_to+2].set(quad_form.Omega)
    
    nonlinear_term = _get_obs_term(observation, p.emission)

    return constants, QuadForm(A=As, b=bs, Omega=Omegas), nonlinear_term

def _expect_quad_form_under_filtering(quad_form, q_filtering):
    return jnp.trace(quad_form.Omega @ quad_form.A @ q_filtering.cov @ quad_form.A.T) + _eval_quad_form(quad_form, q_filtering.mean)

def _expect_V_under_filtering(constants_V, quad_forms, nonlinear_term, q_filtering):
    result = constants_V

    result += jnp.sum(jax.vmap(_expect_quad_form_under_filtering, in_axes=(0, None))(quad_forms, q_filtering)) \
            + _expect_obs_term_under_filtering(nonlinear_term, q_filtering)

    result += 0.5*q_filtering.mean.shape[0]

    return result

def init(observations, p, q):

    constants_V, A, b, Omega, nonlinear_term = _init_V(observations[0], p)
    q_filtering = _init_filtering(observations[0], q.prior, q.emission)
    dim_z = p.transition.cov.shape[0]
    As = jnp.empty(shape=(2*len(observations)-1, dim_z, dim_z))
    bs = jnp.empty(shape=(2*len(observations)-1, dim_z))
    Omegas = jnp.empty_like(As)

    As = As.at[0].set(A)
    bs  = bs.at[0].set(b)
    Omegas = Omegas.at[0].set(Omega)

    return constants_V, QuadForm(A=As, b=bs, Omega=Omegas), nonlinear_term, q_filtering

def _update(observation, integrate_up_to, quad_forms, nonlinear_term, q_filtering, p, q):
    q_backward = _update_backward(q_filtering, q.transition)
    constants, quad_forms, nonlinear_term = _update_V(observation, 
                                                    integrate_up_to,
                                                    quad_forms, 
                                                    nonlinear_term, 
                                                    q_backward, 
                                                    p)
    q_filtering = _update_filtering(observation, q_filtering, q.transition, q.emission)

    return constants, quad_forms, nonlinear_term, q_filtering

def prepare_parameters(model):

    model = actual_model_from_raw_parameters(model)

    transition = Transition(*model.transition, 
                        jnp.linalg.inv(model.transition.cov),
                        jnp.linalg.det(model.transition.cov))
    emission = Emission(*model.emission, 
                        jnp.linalg.inv(model.emission.cov),
                        jnp.linalg.det(model.emission.cov))

    return Model(prior=model.prior, 
                transition=transition, 
                emission=emission)


def linear_gaussian_elbo(p_raw, q_raw, observations):

    p = prepare_parameters(p_raw)
    q = prepare_parameters(q_raw)
    
    constants_V, quad_forms, nonlinear_term, q_filtering = init(observations, p, q)

    integrate_up_to_array = jnp.arange(start=0, stop=2*len(observations[1:]), step=2)

    
    # for observation, integrate_up_to in zip(observations[1:], integrate_up_to_array):
    #     new_constants, quad_forms, nonlinear_term, q_filtering = _update(observation, integrate_up_to, quad_forms, nonlinear_term, q_filtering, p, q)
    #     constants_V += new_constants

    def V_step(carry, x):

        observation, integrate_up_to = x 
        quad_forms, nonlinear_term, q_filtering, p, q = carry 

        new_constants, quad_forms, nonlinear_term, q_filtering = _update(observation, 
                                                                    integrate_up_to,
                                                                    quad_forms, 
                                                                    nonlinear_term, 
                                                                    q_filtering, 
                                                                    p, 
                                                                    q)

        return (quad_forms, nonlinear_term, q_filtering, p, q), new_constants

    (quad_forms, nonlinear_term, q_filtering, p, q), constants = lax.scan(f=V_step, 
                                init=(quad_forms, nonlinear_term, q_filtering, p, q),
                                xs=(observations[1:], integrate_up_to_array))
    constants_V += jnp.sum(constants) 
    constants_V += -_constant_terms_from_log_gaussian(p.transition.cov.shape[0], jnp.linalg.det(q_filtering.cov))

    return _expect_V_under_filtering(constants_V, quad_forms, nonlinear_term, q_filtering)
    

    


        
