from collections import namedtuple
import jax.numpy as jnp

Model = namedtuple('Model',['prior','transition','emission'])
Prior = namedtuple('Prior',['mean','cov'])
Transition = namedtuple('Transition',['weight','bias','cov'])
Emission = namedtuple('Observation',['weight','bias','cov'])
Dims = namedtuple('Dims',['z','x'])
QuadForm = namedtuple('QuadForm',['Omega','A','b'])

def build_covs(model):
    prior_cov = jnp.diag(model.prior.cov ** 2) 
    transition_cov = jnp.diag(model.transition.cov ** 2)
    emission_cov = jnp.diag(model.emission.cov ** 2)
    
    model = Model(prior=Prior(model.prior.mean, prior_cov),
                transition=Transition(model.transition.weight, model.transition.bias, transition_cov),
                emission=Emission(model.emission.weight, model.emission.bias, emission_cov))
                
    return model 
