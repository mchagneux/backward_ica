from jax import vmap, lax, config, numpy as jnp
from collections import namedtuple

from .kalman import filter_step as kalman_filter_step, init as kalman_init
from .misc import parameters_from_raw_parameters

config.update("jax_enable_x64", True)


def _create_quad_form(A, b, Omega):
    return {'A':A, 'b':b, 'Omega':Omega}

def _create_filtering(mean, cov):
    return {'mean':mean, 'cov':cov}

def _create_backward(A, a, cov):
    return {'A':A, 'a':a, 'cov':cov}

def _update_quad_forms_at_index(quad_forms, quad_form, index):
    quad_forms['A'] = quad_forms['A'].at[index].set(quad_form['A'])
    quad_forms['b'] = quad_forms['b'].at[index].set(quad_form['b'])
    quad_forms['Omega'] = quad_forms['Omega'].at[index].set(quad_form['Omega'])
    return quad_forms

def _constant_terms_from_log_gaussian(dim, det_cov):
    return -0.5*(dim * jnp.log(2*jnp.pi) + jnp.log(det_cov))

def _eval_quad_form(quad_form, x):
    common_term = quad_form['A'] @ x + quad_form['b']
    return common_term.T @ quad_form['Omega'] @ common_term

def _expect_obs_term_under_backward(obs_term, q_backward):
    return _expect_quad_form_under_backward(obs_term, q_backward)

def _expect_obs_term_under_filtering(obs_term, q_filtering):
    return _expect_quad_form_under_filtering(obs_term, q_filtering)

def _get_obs_term(observation, p_emission):
    
    return _create_quad_form(A=p_emission['weight'],
                            b=p_emission['bias'] - observation,
                            Omega=- 0.5*p_emission['prec'])

def _init_filtering(observation, q_prior, q_emission):
    return _create_filtering(*kalman_init(observation, q_prior, q_emission)[2:])

def _update_filtering(observation, q_filtering, q_transition, q_emission):
    return _create_filtering(*kalman_filter_step(q_filtering['mean'], q_filtering['cov'], observation, q_transition, q_emission)[2:])

def _update_backward(q_filtering, q_transition):

    filtering_prec = jnp.linalg.inv(q_filtering['cov'])

    backward_prec = q_transition['weight'].T @ q_transition['prec'] @ q_transition['weight'] + filtering_prec

    cov = jnp.linalg.inv(backward_prec)

    common_term = q_transition['weight'].T @ q_transition['prec'] 
    A = cov @ common_term
    a = cov @ (filtering_prec @ q_filtering['mean'] - common_term @  q_transition['bias'])

    return _create_backward(A=A,a=a,cov=cov)

def _integrate_previous_terms(integrate_up_to, quad_forms, nonlinear_term, q_backward):

    quad_forms_transition, quad_forms_emission = quad_forms

    masks = jnp.arange(start=0, stop=quad_forms_transition['A'].shape[0]) <= integrate_up_to
    _expect_quad_forms_under_backward_masked = vmap(_expect_quad_form_under_backward_masked, in_axes=(0,0,None))
    
    constants0, quad_forms_transition = _expect_quad_forms_under_backward_masked(masks, quad_forms_transition, q_backward)

    masks = masks.at[integrate_up_to].set(False)
    constants1, quad_forms_emission = _expect_quad_forms_under_backward_masked(masks, quad_forms_emission, q_backward)

    constants2, quad_form = _expect_obs_term_under_backward(nonlinear_term, q_backward)

    quad_forms_emission = _update_quad_forms_at_index(quad_forms_emission, 
                                                    quad_form, 
                                                    index=integrate_up_to)

    return jnp.sum(constants0) + jnp.sum(constants1) + constants2, quad_forms_transition, quad_forms_emission


def _init_V(observation, p):

    constants_V = _constant_terms_from_log_gaussian(p['transition']['cov'].shape[0], jnp.linalg.det(p['prior']['cov'])) + \
                  _constant_terms_from_log_gaussian(p['emission']['cov'].shape[0], p['emission']['det_cov'])
    
    init_transition_term = {'A':jnp.eye(p['transition']['cov'].shape[0]), 'b':-p['prior']['mean'], 'Omega':-0.5 * jnp.linalg.inv(p['prior']['cov'])}
    nonlinear_term = _get_obs_term(observation, p['emission'])

    return constants_V, init_transition_term, nonlinear_term 

def _expect_transition_quad_form_under_backward(q_backward, p_transition):
    # expectation of the quadratic form that appears in the log of the state transition density

    A=p_transition['weight'] @ q_backward['A'] - jnp.eye(p_transition['cov'].shape[0])
    b=p_transition['weight'] @ q_backward['a'] + p_transition['bias']
    Omega = -0.5*p_transition['prec']

    return {'A':A, 'b':b, 'Omega':Omega} 

def _no_integration(quad_form, q_backward):
    return 0.0, quad_form

def _expect_quad_form_under_backward(quad_form, q_backward):
    Sigma = quad_form['A'] @ q_backward['cov'] @ quad_form['A'].T
    constant = jnp.trace(quad_form['Omega'] @ Sigma)
    A = quad_form['A'] @ q_backward['A']
    b = quad_form['A'] @ q_backward['a'] + quad_form['b']
    Omega = quad_form['Omega']
    return constant, {'A':A, 'b':b, 'Omega':Omega}

def _expect_quad_form_under_backward_masked(mask, quad_form, q_backward):
    # if mask: 
    #     return _expect_quad_form_under_backward(quad_form, q_backward)

    # return _no_integration(quad_form, q_backward)
    return lax.cond(mask, _expect_quad_form_under_backward, _no_integration, quad_form, q_backward)

def _update_V(observation, integrate_up_to, quad_forms, nonlinear_term, q_backward, p):

    constants, quad_forms_transition, quad_forms_emission =  _integrate_previous_terms(integrate_up_to, quad_forms, nonlinear_term, q_backward)

    dim_z, dim_x = p['transition']['cov'].shape[0], p['emission']['cov'].shape[0]
    constants +=  _constant_terms_from_log_gaussian(dim_x, p['emission']['det_cov']) \
                +   _constant_terms_from_log_gaussian(dim_z, p['transition']['det_cov']) \
                + - _constant_terms_from_log_gaussian(dim_z, jnp.linalg.det(q_backward['cov'])) \
                +  0.5 * dim_z \
                + - 0.5 * jnp.trace(p['transition']['prec'] @ p['transition']['weight'] @ q_backward['cov'] @ p['transition']['weight'].T)

    quad_form = _expect_transition_quad_form_under_backward(q_backward, p['transition'])


    quad_forms_transition = _update_quad_forms_at_index(quad_forms_transition, 
                                                    quad_form, 
                                                    index=integrate_up_to+1)
    
    nonlinear_term = _get_obs_term(observation, p['emission'])

    return constants, [quad_forms_transition, quad_forms_emission], nonlinear_term

def _expect_quad_form_under_filtering(quad_form, q_filtering):
    return jnp.trace(quad_form['Omega'] @ quad_form['A'] @ q_filtering['cov'] @ quad_form['A'].T) + _eval_quad_form(quad_form, q_filtering['mean'])

def _expect_V_under_filtering(constants_V, quad_forms, nonlinear_term, q_filtering):
    result = constants_V

    quad_forms_transition, quad_forms_emission = quad_forms
    quad_forms_emission['A'] = quad_forms_emission['A'][:-1]
    quad_forms_emission['b'] = quad_forms_emission['b'][:-1]
    quad_forms_emission['Omega'] = quad_forms_emission['Omega'][:-1]

    expect_quad_forms_under_filtering = vmap(_expect_quad_form_under_filtering, in_axes=(0, None))

    result += jnp.sum(expect_quad_forms_under_filtering(quad_forms_transition, q_filtering)) \
            + jnp.sum(expect_quad_forms_under_filtering(quad_forms_emission, q_filtering)) \
            + _expect_obs_term_under_filtering(nonlinear_term, q_filtering)

    result += 0.5*q_filtering['mean'].shape[0]

    return result

def init(observations, p, q):

    constants_V, init_transition_term, nonlinear_term = _init_V(observations[0], p)
    q_filtering = _init_filtering(observations[0], q['prior'], q['emission'])
    dim_z = p['transition']['cov'].shape[0]

    num_terms = len(observations)
    As = jnp.empty(shape=(num_terms, dim_z, dim_z))
    bs = jnp.empty(shape=(num_terms, dim_z))
    Omegas = jnp.empty(shape=(num_terms, dim_z, dim_z))

    As = As.at[0].set(init_transition_term['A'])
    bs  = bs.at[0].set(init_transition_term['b'])
    Omegas = Omegas.at[0].set(init_transition_term['Omega'])

    quad_forms_transition = _create_quad_form(A=As, b=bs, Omega=Omegas)

    dim_x = p['emission']['cov'].shape[0]

    quad_forms_emission = _create_quad_form(A=jnp.empty(shape=(num_terms, dim_x, dim_z)),
                                            b=jnp.empty(shape=(num_terms, dim_x)),
                                            Omega=jnp.empty(shape=(num_terms, dim_x, dim_x)))
            
    quad_forms = [quad_forms_transition, quad_forms_emission]
    return constants_V, quad_forms, nonlinear_term, q_filtering

def _update(observation, integrate_up_to, quad_forms, nonlinear_term, q_filtering, p, q):
    q_backward = _update_backward(q_filtering, q['transition'])
    constants, quad_forms, nonlinear_term = _update_V(observation, 
                                                    integrate_up_to,
                                                    quad_forms, 
                                                    nonlinear_term, 
                                                    q_backward, 
                                                    p)

    q_filtering = _update_filtering(observation, q_filtering, q['transition'], q['emission'])

    return constants, quad_forms, nonlinear_term, q_filtering

def prepare_parameters(model):

    model = parameters_from_raw_parameters(model)

    model['transition']['prec'] =  jnp.linalg.inv(model['transition']['cov'])
    model['transition']['det_cov'] = jnp.linalg.det(model['transition']['cov'])

    model['emission']['prec'] =  jnp.linalg.inv(model['emission']['cov'])
    model['emission']['det_cov'] = jnp.linalg.det(model['emission']['cov'])


    return model

def linear_gaussian_elbo(p_raw, q_raw, observations):

    p = prepare_parameters(p_raw)
    q = prepare_parameters(q_raw)
    
    constants_V, quad_forms, nonlinear_term, q_filtering = init(observations, p, q)

    observations = observations[1:]

    integrate_up_to_array = jnp.arange(start=0, stop=len(observations))

    
    # for observation, integrate_up_to in zip(observations, integrate_up_to_array):
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
                                xs=(observations, integrate_up_to_array))

    constants_V += jnp.sum(constants) 
    constants_V += -_constant_terms_from_log_gaussian(p['transition']['cov'].shape[0], jnp.linalg.det(q_filtering['cov']))

    return _expect_V_under_filtering(constants_V, quad_forms, nonlinear_term, q_filtering)
    

    


        
