from collections import namedtuple
from dataclasses import dataclass
from queue import PriorityQueue
from typing import Any
from jax import disable_jit, numpy as jnp, vmap, config, random, lax, jit
from functools import partial
from jax.tree_util import register_pytree_node_class, tree_multimap, tree_map
from jax.scipy.linalg import solve_triangular, cho_solve, cho_factor
import matplotlib.pyplot as plt
from tqdm import tqdm
config.update('jax_enable_x64',True)
import numpy as np 
# Containers for parameters of various objects 

GaussianKernelBaseParams = namedtuple('GaussianKernelBaseParams', ['map_params', 'cov_base'])
GaussianKernelParams = namedtuple('GaussianKernelParams', ['map_params','cov_chol','cov','prec','log_det'])

LinearGaussianKernelBaseParams = namedtuple('LinearGaussianKernelBaseParams',['matrix', 'bias', 'cov_base'])
LinearGaussianKernelParams = namedtuple('LinearGaussianKernelParams',['matrix', 'bias', 'cov_chol', 'cov','prec','log_det'])

GaussianBaseParams = namedtuple('GaussianBaseParams', ['mean', 'cov_base'])
GaussianParams = namedtuple('GaussianParams', ['mean', 'cov_chol','cov','prec','log_det'])

HMMParams = namedtuple('HMMParams',['prior','transition','emission'])

NeuralSmootherParams = namedtuple('NeuralSmootherParams', ['prior', 'shared', 'filt_predict', 'filt_update', 'backwd_update'])



def tree_prepend(prep, tree):
    preprended = tree_multimap(
        lambda a, b: jnp.concatenate((a[None,:], b)), prep, tree
    )
    return preprended


def tree_append(tree, app):
    appended = tree_multimap(
        lambda a, b: jnp.concatenate((a, b[None,:])), tree, app
    )
    return appended

def tree_droplast(tree):
    '''Drop last index from each leaf'''
    return tree_map(lambda a: a[:-1], tree)


def tree_dropfirst(tree):
    '''Drop first index from each leaf'''
    return tree_map(lambda a: a[1:], tree)


def tree_get_idx(idx, tree):
    '''Get idx row from each leaf of tuple'''
    return tree_map(lambda a: a[idx], tree)

def tree_get_slice(start, stop, tree):
    '''Get idx row from each leaf of tuple'''
    return tree_map(lambda a: a[start:stop], tree)

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


def plot_training_curves(avg_elbos, avg_evidence=None):


    num_fits = len(avg_elbos)
    fig, axes = plt.subplots(1,num_fits, sharey=True)
    for fit_nb in range(num_fits):
        axes[fit_nb].plot(avg_elbos[fit_nb], label='$\mathcal{L}(\\theta,\\phi)$')
        if avg_evidence is not None:
            axes[fit_nb].axhline(y=avg_evidence, c='red', label = '$log p_{\\theta}(x)$')
        axes[fit_nb].set_xlabel('Epoch') 
        axes[fit_nb].set_title(f'Fit {fit_nb+1}')
        axes[fit_nb].legend()

    plt.show()

def plot_example_smoothed_states(p, q, theta, phi, state_seqs, obs_seqs, seq_nb, *args):

    fig, (ax0, ax1) = plt.subplots(1,2, sharey=True)
    plot_relative_errors_1D(ax0, state_seqs[seq_nb], *p.smooth_seq(obs_seqs[seq_nb], theta, *args))
    ax0.set_title('True params')

    plot_relative_errors_1D(ax1, state_seqs[seq_nb], *q.smooth_seq(obs_seqs[seq_nb], phi, *args))
    ax1.set_title('Fitted params')

    plt.tight_layout()
    plt.autoscale(True)
    plt.show()



def plot_fit_results_1D(q, q_params, state_seqs, obs_seqs, avg_elbos, avg_evidence, seq_nb, *aux):
    fig = plt.figure(figsize=(15,5))

    ax0 = fig.add_subplot(131)
    ax0.plot(avg_elbos, label='$\mathcal{L}(\\theta,\\phi)$')
    ax0.axhline(y=avg_evidence, c='red', label = '$log p_{\\theta}(x)$')

    ax0.set_xlabel('Epoch') 
    ax0.set_title('Training')
    ax0.legend()

    ax1 = fig.add_subplot(132)
    plot_relative_errors_1D(ax1, state_seqs[seq_nb], *q.smooth_seq(obs_seqs[seq_nb], q_params))
    ax1.set_title('Example sequence backward variational')

    ax2 = fig.add_subplot(133)
    ax2.set_title('Associated observations')
    ax2.plot(obs_seqs[seq_nb], marker='.', linestyle='dotted', label='x')
    ax2.set_xlabel('t')

    plt.tight_layout()
    plt.autoscale(True)
    plt.legend()
    plt.show()


def smoothing_results_mse_with_aux(state_seqs, obs_seqs, smoother, params, *aux):
    prior_keys, resampling_keys, proposal_keys, num_particles = aux
    squared_error_on_seq = vmap(lambda state_seq, obs_seq, prior_keys, resampling_keys, proposal_keys: (smoother.smooth_sum_of_means(obs_seq, params, prior_keys, resampling_keys, proposal_keys, num_particles) - jnp.sum(state_seq))**2)
    return jnp.sum(squared_error_on_seq(state_seqs, obs_seqs, prior_keys, resampling_keys, proposal_keys)) / (state_seqs.shape[0] * state_seqs.shape[1])


def smoothing_results_mse(state_seqs, obs_seqs, smoother, params):
    squared_error_on_seq = vmap(lambda state_seq, obs_seq: (jnp.abs(smoother.smooth_sum_of_means(obs_seq, params) - jnp.sum(state_seq, axis=0))))
    return jnp.sum(squared_error_on_seq(state_seqs, obs_seqs)) / (state_seqs.shape[0] * state_seqs.shape[1])

def smoothing_results_mse_different_lengths(state_seqs, obs_seqs, smoother, params, step):
    
    additive = []
    for length in range(2, state_seqs.shape[1], step):
        additive.append(smoothing_results_mse(state_seqs[:,:length,:], obs_seqs[:,:length,:], smoother, params))


    return additive

def compare_mse_for_different_lengths(q, q_params, state_seqs, obs_seqs, step=4):
    results_fitted_params = smoothing_results_mse_different_lengths(state_seqs, obs_seqs, q, q_params, step)
    seq_lengths = np.arange(2, state_seqs.shape[1], step)
    plt.plot(seq_lengths, results_fitted_params, c='b', label='Backward variational', marker='.', linestyle='dotted')
    plt.xlabel('Sequence length')
    plt.ylabel('MSE between smoothed means and true states')
    plt.legend()
    plt.show()    


def plot_smoothing_wrt_seq_length_linear(key, ref_smoother, approx_smoother, ref_params, approx_params, seq_length, step, ref_smoother_name, approx_smoother_name):
    timesteps = range(2, seq_length, step)

    compute_ref_filt_seq = lambda obs_seq: ref_smoother.compute_filt_seq(obs_seq, ref_params)
    compute_ref_backwd_seq = lambda filt_seq: ref_smoother.compute_backwd_seq(filt_seq, ref_params)
    compute_approx_filt_seq = lambda obs_seq: approx_smoother.compute_filt_seq(obs_seq, approx_params)
    compute_approx_backwd_seq = lambda filt_seq: approx_smoother.compute_backwd_seq(filt_seq, approx_params)
    
    ref_backwd_pass = ref_smoother.backwd_pass
    approx_backwd_pass = approx_smoother.backwd_pass

    def results_for_single_seq(state_seq, obs_seq):

        ref_filt_seq, approx_filt_seq = compute_ref_filt_seq(obs_seq), compute_approx_filt_seq(obs_seq)
        ref_backwd_seq, approx_backwd_seq = compute_ref_backwd_seq(ref_filt_seq), compute_approx_backwd_seq(approx_filt_seq)
        kalman_wrt_states, vi_wrt_states, vi_vs_kalman = [], [], []

        def result_up_to_length(length):

            ref_smoothed_means = ref_backwd_pass(tree_get_idx(length, ref_filt_seq), tree_get_slice(0,length-1, ref_backwd_seq))[0]
            approx_smoothed_means = approx_backwd_pass(tree_get_idx(length, approx_filt_seq), tree_get_slice(0,length-1, approx_backwd_seq))[0]
            
            kalman_wrt_states = jnp.abs(jnp.sum(ref_smoothed_means - state_seq[:length], axis=0))
            vi_wrt_states = jnp.abs(jnp.sum(approx_smoothed_means - state_seq[:length], axis=0))
            vi_vs_kalman = jnp.abs(jnp.sum(approx_smoothed_means - ref_smoothed_means, axis=0))

            return kalman_wrt_states, vi_wrt_states, vi_vs_kalman
        
        for length in timesteps: 
            result = result_up_to_length(length)
            kalman_wrt_states.append(result[0])
            vi_wrt_states.append(result[1])
            vi_vs_kalman.append(result[2])

        ref_smoothed_means = ref_backwd_pass(tree_get_idx(-1, ref_filt_seq), tree_get_slice(0,-1, ref_backwd_seq))[0]
        approx_smoothed_means = approx_backwd_pass(tree_get_idx(-1, approx_filt_seq), tree_get_slice(0,-1, approx_backwd_seq))[0]
        vi_vs_kalman_marginals = jnp.abs(ref_smoothed_means - approx_smoothed_means)[jnp.array(timesteps)]

        return kalman_wrt_states, vi_wrt_states, vi_vs_kalman, vi_vs_kalman_marginals


    state_seqs, obs_seqs = vmap(ref_smoother.sample_seq, in_axes=(0,None,None))(random.split(key, 5), ref_params, seq_length)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4)

    
    for seq_nb, (state_seq, obs_seq) in tqdm(enumerate(zip(state_seqs, obs_seqs))):
        kalman_wrt_states, vi_wrt_states, vi_vs_kalman, vi_vs_kalman_marginals = results_for_single_seq(state_seq, obs_seq)
        ax0.plot(timesteps, kalman_wrt_states, label = f'Sequence {seq_nb}')
        ax1.plot(timesteps, vi_wrt_states, label = f'Sequence {seq_nb}')
        ax2.plot(timesteps, vi_vs_kalman, label = f'Sequence {seq_nb}')
        ax3.plot(timesteps, vi_vs_kalman_marginals, label = f'Sequence {seq_nb}')

    
    ax0.set_title(f'{ref_smoother_name} vs states (additive)')
    ax0.set_xlabel('Sequence length')
    ax0.legend()

    ax1.set_title(f'{approx_smoother_name} vs states (additive)')
    ax1.set_xlabel('Sequence length')
    ax1.legend()


    ax2.set_title(f'{approx_smoother_name} vs {ref_smoother_name} (additive)')
    ax2.set_xlabel('Sequence length')
    ax2.legend()


    ax3.set_title(f'{approx_smoother_name} vs {ref_smoother_name} (marginals)')
    ax3.set_xlabel('Sequence length')
    ax3.legend()


    plt.show()


def plot_smoothing_wrt_seq_length_nonlinear(key, ref_smoother, approx_smoother, ref_params, approx_params, seq_length, step, ref_smoother_name, approx_smoother_name, prior_keys, resampling_keys, proposal_keys, backwd_sampling_keys):
    timesteps = range(2, seq_length, step)

    compute_ref_filt_seq = lambda obs_seq: ref_smoother.compute_filt_seq(obs_seq, ref_params)
    compute_approx_filt_seq = lambda obs_seq: approx_smoother.compute_filt_seq(obs_seq, approx_params)
    compute_approx_backwd_seq = lambda filt_seq: approx_smoother.compute_backwd_seq(filt_seq, approx_params)
    
    approx_backwd_pass = approx_smoother.backwd_pass

    def results_for_single_seq(state_seq, obs_seq):

        ref_filt_seq, approx_filt_seq = compute_ref_filt_seq(obs_seq), compute_approx_filt_seq(obs_seq)
        ref_backwd_seq, approx_backwd_seq = compute_ref_backwd_seq(ref_filt_seq), compute_approx_backwd_seq(approx_filt_seq)
        kalman_wrt_states, vi_wrt_states, vi_vs_kalman = [], [], []

        def result_up_to_length(length):

            ref_smoothed_means = ref_backwd_pass(tree_get_idx(length, ref_filt_seq), tree_get_slice(0,length-1, ref_backwd_seq))[0]
            approx_smoothed_means = approx_backwd_pass(tree_get_idx(length, approx_filt_seq), tree_get_slice(0,length-1, approx_backwd_seq))[0]
            
            kalman_wrt_states = jnp.abs(jnp.sum(ref_smoothed_means - state_seq[:length], axis=0))
            vi_wrt_states = jnp.abs(jnp.sum(approx_smoothed_means - state_seq[:length], axis=0))
            vi_vs_kalman = jnp.abs(jnp.sum(approx_smoothed_means - ref_smoothed_means, axis=0))

            return kalman_wrt_states, vi_wrt_states, vi_vs_kalman
        
        for length in timesteps: 
            result = result_up_to_length(length)
            kalman_wrt_states.append(result[0])
            vi_wrt_states.append(result[1])
            vi_vs_kalman.append(result[2])

        ref_smoothed_means = ref_backwd_pass(tree_get_idx(-1, ref_filt_seq), tree_get_slice(0,-1, ref_backwd_seq))[0]
        approx_smoothed_means = approx_backwd_pass(tree_get_idx(-1, approx_filt_seq), tree_get_slice(0,-1, approx_backwd_seq))[0]
        vi_vs_kalman_marginals = jnp.abs(ref_smoothed_means - approx_smoothed_means)[jnp.array(timesteps)]

        return kalman_wrt_states, vi_wrt_states, vi_vs_kalman, vi_vs_kalman_marginals


    state_seqs, obs_seqs = vmap(ref_smoother.sample_seq, in_axes=(0,None,None))(random.split(key, 5), ref_params, seq_length)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4)

    
    for seq_nb, (state_seq, obs_seq) in tqdm(enumerate(zip(state_seqs, obs_seqs))):
        kalman_wrt_states, vi_wrt_states, vi_vs_kalman, vi_vs_kalman_marginals = results_for_single_seq(state_seq, obs_seq)
        ax0.plot(timesteps, kalman_wrt_states, label = f'Sequence {seq_nb}')
        ax1.plot(timesteps, vi_wrt_states, label = f'Sequence {seq_nb}')
        ax2.plot(timesteps, vi_vs_kalman, label = f'Sequence {seq_nb}')
        ax3.plot(timesteps, vi_vs_kalman_marginals, label = f'Sequence {seq_nb}')

    
    ax0.set_title(f'{ref_smoother_name} vs states (additive)')
    ax0.set_xlabel('Sequence length')
    ax0.legend()

    ax1.set_title(f'{approx_smoother_name} vs states (additive)')
    ax1.set_xlabel('Sequence length')
    ax1.legend()


    ax2.set_title(f'{approx_smoother_name} vs {ref_smoother_name} (additive)')
    ax2.set_xlabel('Sequence length')
    ax2.legend()


    ax3.set_title(f'{approx_smoother_name} vs {ref_smoother_name} (marginals)')
    ax3.set_xlabel('Sequence length')
    ax3.legend()


    plt.show()



