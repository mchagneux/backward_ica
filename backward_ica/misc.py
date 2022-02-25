import jax.numpy as jnp

from collections import namedtuple



def parameters_from_raw_parameters(raw_model):

    model = raw_model
    model['prior']['cov'] = jnp.diag(raw_model['prior']['cov'] ** 2) 
    model['transition']['weight'] = jnp.diag(model['transition']['weight'])

    model['transition']['cov'] = jnp.diag(raw_model['transition']['cov'] ** 2)
    model['emission']['cov'] = jnp.diag(raw_model['emission']['cov'] ** 2)
                    
    return model
