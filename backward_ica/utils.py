from collections import namedtuple
from dataclasses import dataclass 
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class 

# Containers for parameters of various objects 
GaussianKernelBaseParams = namedtuple('GaussianKernelParams', ['map_params', 'cov_base'])
GaussianKernelParams = namedtuple('GaussianKernelParams', ['map_params', 'cov_base', 'cov', 'prec', 'det'])

LinearGaussianKernelBaseParams = namedtuple('LinearGaussianKernelParams',['matrix', 'bias', 'cov_base'])
LinearGaussianKernelParams = namedtuple('LinearGaussianKernelParams',['matrix', 'bias', 'cov_base', 'cov', 'prec', 'det'])

GaussianBaseParams = namedtuple('GaussianParams', ['mean', 'cov_base'])
GaussianParams = namedtuple('GaussianParams', ['mean', 'cov_base', 'cov', 'prec', 'det'])

HMMParams = namedtuple('HMMParams',['prior','transition','emission'])


@dataclass(init=True)
@register_pytree_node_class
class QuadTerm:

    W: jnp.ndarray
    v: jnp.ndarray
    c: jnp.ndarray

    def __iter__(self):
        return iter((self.W, self.v, self.c))

    def __add__(self, other):
        return QuadTerm(W = self.W + other.W, 
                        v = self.v + other.v, 
                        c = self.c + other.c)

    def __rmul__(self, other):
        return QuadTerm(W=other*self.W, 
                        v=other*self.v, 
                        c=other*self.c) 
    def evaluate(self, x):
        return x.T @ self.W @ x + self.v.T @ x + self.c

    def tree_flatten(self):
        return ((self.W, self.v, self.c), None) 

    @staticmethod
    def from_A_b_Omega(A, b, Omega):
        return QuadTerm(W = A.T @ Omega @ A, 
                        v = A.T @ (Omega + Omega.T) @ b, 
                        c = b.T @ Omega @ b)
    @staticmethod 
    def evaluate_from_A_b_Omega(A, b, Omega, x):
        common_term = A @ x + b 
        return common_term.T @ Omega @ common_term



    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)