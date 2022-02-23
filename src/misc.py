from collections import namedtuple

Model = namedtuple('Model',['prior','transition','emission'])
Prior = namedtuple('Prior',['mean','cov'])
Transition = namedtuple('Transition',['weight','bias','cov'])
Emission = namedtuple('Observation',['weight','bias','cov'])
Dims = namedtuple('Dims',['z','x'])
QuadForm = namedtuple('QuadForm',['Omega','A','b'])

