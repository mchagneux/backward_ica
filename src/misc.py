from collections import namedtuple
import jax.numpy as jnp

Model = namedtuple('Model',['prior','transition','emission'])
Prior = namedtuple('Prior',['mean','cov'])
Transition = namedtuple('Transition',['weight','bias','cov'])
Emission = namedtuple('Observation',['weight','bias','cov'])
Dims = namedtuple('Dims',['z','x'])
QuadForm = namedtuple('QuadForm',['Omega','A','b'])

def actual_model_from_raw_parameters(raw_model):

    prior_cov = jnp.diag(raw_model.prior.cov ** 2) 
    transition_cov = jnp.diag(raw_model.transition.cov ** 2)
    emission_cov = jnp.diag(raw_model.emission.cov ** 2)
                    
    return Model(prior=Prior(raw_model.prior.mean, prior_cov),
                transition=Transition(jnp.diag(raw_model.transition.weight), raw_model.transition.bias, transition_cov),
                emission=Emission(jnp.diag(raw_model.emission.weight), raw_model.emission.bias, emission_cov)) 
