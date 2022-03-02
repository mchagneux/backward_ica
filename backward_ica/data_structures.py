from typing import *
from dataclasses import dataclass

from jax.tree_util import register_pytree_node_class
import jax.numpy as jnp


@dataclass(init=True, repr=True)
@register_pytree_node_class
class QuadForm:

    A:jnp.ndarray
    b:jnp.ndarray
    Omega:jnp.ndarray 

    def __iter__(self):
        return iter((self.A, self.b, self.Omega))
    
    def tree_flatten(self):
        return ((self.A, self.b, self.Omega), None) 

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@dataclass(init=True, repr=True)
@register_pytree_node_class
class Backward:

    A:jnp.ndarray
    a:jnp.ndarray
    cov:jnp.ndarray 

    def __iter__(self):
        return iter((self.A, self.a, self.cov))
    
    def tree_flatten(self):
        return ((self.A, self.a, self.cov), None) 

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@dataclass(init=True, repr=True)
@register_pytree_node_class
class Filtering:

    mean:jnp.ndarray
    cov:jnp.ndarray
    
    def __iter__(self):
        return iter((self.mean, self.cov))
        
    def tree_flatten(self):
        return ((self.mean, self.cov), None) 

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)
