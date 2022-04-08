from collections import namedtuple
from dataclasses import dataclass
from queue import PriorityQueue
from typing import Any
from jax import numpy as jnp, vmap, config
from jax.tree_util import register_pytree_node_class 
from jax.scipy.linalg import solve_triangular, cho_solve, cho_factor
import matplotlib.pyplot as plt
config.update('jax_enable_x64',True)
# Containers for parameters of various objects 

GaussianKernelBaseParams = namedtuple('GaussianKernelBaseParams', ['map_params', 'cov_base'])
GaussianKernelParams = namedtuple('GaussianKernelParams', ['map_params','cov_chol','cov','prec','log_det'])

LinearGaussianKernelBaseParams = namedtuple('LinearGaussianKernelBaseParams',['matrix', 'bias', 'cov_base'])
LinearGaussianKernelParams = namedtuple('LinearGaussianKernelParams',['matrix', 'bias', 'cov_chol', 'cov','prec','log_det'])

GaussianBaseParams = namedtuple('GaussianBaseParams', ['mean', 'cov_base'])
GaussianParams = namedtuple('GaussianParams', ['mean', 'cov_chol','cov','prec','log_det'])

HMMParams = namedtuple('HMMParams',['prior','transition','emission'])

def chol_from_inv(mat):
    tril_inv = jnp.swapaxes(
        jnp.linalg.cholesky(mat[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
    )
    identity = jnp.broadcast_to(jnp.identity(mat.shape[-1]), tril_inv.shape)
    return solve_triangular(tril_inv, identity, lower=True)

def inv(mat):
    return cho_solve(c_and_lower=cho_factor(mat, True), 
                    b=jnp.eye(mat.shape[0]))

def inv_of_chol(mat):
    return inv_of_chol_from_chol(jnp.linalg.cholesky(mat))

def inv_of_chol_from_chol(mat_chol):
    return solve_triangular(a=mat_chol, b=jnp.eye(mat_chol.shape[0]), lower=True)

def inv_from_chol(mat_chol):
    return cho_solve(c_and_lower=(mat_chol,True), 
                b=jnp.eye(mat_chol.shape[0]))

def log_det_from_chol(chol):
    return jnp.sum(jnp.log(jnp.diagonal(chol)**2))


def cov_params_from_cov_chol(cov_chol):
    cov = cov_chol @ cov_chol.T 
    return cov_chol, cov, inv_from_chol(cov_chol), log_det_from_chol(cov_chol)

def cov_params_from_cov(cov):
    cov_chol = jnp.linalg.cholesky(cov)
    return cov_chol, cov, inv_from_chol(cov_chol), log_det_from_chol(cov_chol)

# @dataclass(frozen=True, init=True)
# @register_pytree_node_class
# class HMMParams:
#     prior:Any
#     transition:Any 
#     emission:Any

#     def tree_flatten(self):
#         return ((self.prior, self.transition, self.emission), None) 

#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*children)

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


def plot_fit_results_1D(p, q, p_params, q_params, state_seqs, obs_seqs, avg_elbos, avg_evidence, seq_nb=0):
    fig = plt.figure(figsize=(10,10))

    ax0 = fig.add_subplot(131)
    ax0.plot(avg_elbos, label='$\mathcal{L}(\\theta,\\phi)$')
    ax0.axhline(y=avg_evidence, c='red', label = '$log p_{\\theta}(x)$' )
    ax0.set_xlabel('Epoch') 

    ax1 = fig.add_subplot(132)
    plot_relative_errors_1D(ax1, state_seqs[seq_nb], *p.smooth_seq(obs_seqs[seq_nb], p_params))
    ax1.set_title('Kalman')

    ax2 = fig.add_subplot(133, sharey=ax1)
    plot_relative_errors_1D(ax2, state_seqs[seq_nb], *q.smooth_seq(obs_seqs[seq_nb], q_params))
    ax2.set_title('Backward variational')

    plt.autoscale(True)
    plt.legend()
    plt.show()

def mse_smoothed_means_against_true_states(state_seqs, obs_seqs, smoother, params):
    v_smoother = vmap(lambda seq: smoother.smooth_seq(seq, params)[0])
    smoothed_means = v_smoother(obs_seqs)
    return jnp.mean((smoothed_means - state_seqs)**2)



# if __name__ == '__main__':
#     import jax 
#     
#     key = jax.random.PRNGKey(0)
#     d = 3
#     tril_values = jax.random.uniform(key, shape=(d*(d+1) // 2,))
#     A_chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(tril_values)
#     A = A_chol @ A_chol.T 

#     print(cholesky_of_inverse(jnp.linalg.inv(A)) - A_chol)


