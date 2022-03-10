from copy import deepcopy 
import jax.numpy as jnp


def increase_parameterization(raw_model):
    model = deepcopy(raw_model)

    model['transition']['weight'] = jnp.diag(model['transition']['weight'])
    return model 
    
def format_p(p_raw):

    model = deepcopy(p_raw)
    
    model['prior']['cov'] = jnp.diag(model['prior']['cov'] ** 2) 
    model['transition']['params']['weight'] = jnp.diag(model['transition']['params']['weight'])

    model['transition']['cov'] = jnp.diag(model['transition']['cov'] ** 2)
    model['emission']['cov'] = jnp.diag(model['emission']['cov'] ** 2)
                    
    return model

def format_q(q_raw):

    model = deepcopy(q_raw)
    
    model['prior']['cov'] = jnp.diag(model['prior']['cov'] ** 2) 
    model['transition']['cov'] = jnp.diag(model['transition']['cov'] ** 2)
    model['emission']['cov'] = jnp.diag(model['emission']['cov'] ** 2)
                    
    return model
