import jax.numpy as jnp
from collections import namedtuple

Model = namedtuple('Model',['prior','transition','emission'])
Prior = namedtuple('Prior',['mean','cov'])
Transition = namedtuple('Transition',['matrix','offset','cov'])
Emission = namedtuple('Observation',['matrix','offset','cov'])
Dims = namedtuple('Dims',['z','x'])

QuadForm = namedtuple('QuadForm',['Omega','A','b'])

