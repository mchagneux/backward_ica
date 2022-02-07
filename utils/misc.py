import jax.numpy as jnp
from jax import jit 
from collections import namedtuple

PriorParams = namedtuple('prior',['mean','cov'])
TransitionParams = namedtuple('transition',['matrix','offset','cov'])
ObservationParams = namedtuple('observation',['matrix','offset','cov'])

ModelParams = namedtuple('model',['transition','observation','prior'])
QuadForm = namedtuple('quad_form',['Omega','A','b'])
Dims = namedtuple('Dims',['z','x'])

# class QuadForm:
#     def __init__(self, Omega=None, A=None, b=None, shape=None, name=None):

#         if name is not None: self.name = name
#         if b is not None: 
#             self.b = b 
#             self.shape = b.shape 
#         else: 
#             self.shape = shape
#             self.b = np.zeros(shape)

#         self.Omega = np.eye(self.shape[0], self.shape[0]) if Omega is None else Omega
#         self.A = np.eye(self.shape[0], self.shape[0]) if A is None else A


#     def __call__(self, u):
#         u = u.reshape(self.b.shape)
#         result = (self.A @ u + self.b).T @ self.Omega @ (self.A @ u + self.b)
#         return result.squeeze()




