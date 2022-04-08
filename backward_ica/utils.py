from collections import namedtuple
from dataclasses import dataclass
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class 
import matplotlib.pyplot as plt

# Containers for parameters of various objects 

GaussianKernelBaseParams = namedtuple('GaussianKernelBaseParams', ['map_params', 'cov_base'])
GaussianKernelParams = namedtuple('GaussianKernelParams', ['map_params','cov_chol','cov','prec','log_det'])

LinearGaussianKernelBaseParams = namedtuple('LinearGaussianKernelBaseParams',['matrix', 'bias', 'cov_base'])
LinearGaussianKernelParams = namedtuple('LinearGaussianKernelParams',['matrix', 'bias', 'cov_chol', 'cov','prec','log_det'])

GaussianBaseParams = namedtuple('GaussianBaseParams', ['mean', 'cov_base'])
GaussianParams = namedtuple('GaussianParams', ['mean', 'cov_chol','cov','prec','log_det'])

HMMParams = namedtuple('HMMParams',['prior','transition','emission'])

def log_det_from_chol(chol):
    return jnp.sum(jnp.log(jnp.diagonal(chol)**2))

def prec_from_chol(chol):
    inv_chol = jnp.linalg.inv(chol)
    return inv_chol.T @ inv_chol

def cov_params_from_cov_chol(cov_chol):
    cov = cov_chol @ cov_chol.T 
    return cov_chol, cov, prec_from_chol(cov_chol), log_det_from_chol(cov_chol)

def cov_params_from_cov(cov):
    cov_chol = jnp.linalg.cholesky(cov)
    return cov_chol, cov, prec_from_chol(cov_chol), log_det_from_chol(cov_chol)

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


def plot_relative_errors_1D(ax, true_sequence, pred_means, pred_covs):
    true_sequence, pred_means, pred_covs = true_sequence.squeeze(), pred_means.squeeze(), pred_covs.squeeze()
    time_axis = range(len(true_sequence))
    ax.errorbar(x=time_axis, fmt = '_', y=pred_means, yerr=1.96 * jnp.sqrt(pred_covs), label='Smoothed z, $1.96\\sigma$')
    ax.scatter(x=time_axis, marker = '_', y=true_sequence, c='r', label='True z')
    ax.set_xlabel('t')



