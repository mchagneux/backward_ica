from collections import namedtuple
from dataclasses import dataclass
from distutils import log
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
import json
import os 
import pickle 
import argparse
# Containers for parameters of various objects 

GaussianKernelBaseParams = namedtuple('GaussianKernelBaseParams', ['map_params', 'cov_base'])
GaussianKernelParams = namedtuple('GaussianKernelParams', ['map_params','cov_chol','cov','prec','log_det'])

LinearGaussianKernelBaseParams = namedtuple('LinearGaussianKernelBaseParams',['matrix', 'bias', 'cov_base'])
LinearGaussianKernelParams = namedtuple('LinearGaussianKernelParams',['matrix', 'bias', 'cov_chol', 'cov','prec','log_det'])

GaussianBaseParams = namedtuple('GaussianBaseParams', ['mean', 'cov_base'])
GaussianParams = namedtuple('GaussianParams', ['mean', 'cov_chol','cov','prec','log_det'])

HMMParams = namedtuple('HMMParams',['prior','transition','emission'])

NeuralSmootherParams = namedtuple('NeuralSmootherParams', ['prior', 'shared', 'filt_predict', 'filt_update', 'backwd_update'])

FiltParams = namedtuple('FiltParams', ['state', 'mean', 'cov', 'log_det'])
BackwardParams = namedtuple('BackwardParams', ['matrix', 'bias', 'cov', 'log_det'])


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


def log_det_from_cov(cov):
    return log_det_from_chol(jnp.linalg.cholesky(cov))

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
    true_sequence, pred_means, pred_covs = true_sequence.squeeze()[:64], pred_means.squeeze()[:64], pred_covs.squeeze()[:64]
    time_axis = range(len(true_sequence))[:64]
    ax.errorbar(x=time_axis, fmt = '_', y=pred_means, yerr=1.96 * jnp.sqrt(pred_covs), label='Smoothed z, $1.96\\sigma$')
    ax.scatter(x=time_axis, marker = '_', y=true_sequence, c='r', label='True z')
    ax.set_xlabel('t')

def save_args(args, name, save_dir):
    with open(os.path.join(save_dir, f'{name}.json'), 'w') as f:
        args_dict = vars(args)
        json.dump(args_dict, f)

def load_args(name, save_dir):
    with open(os.path.join(save_dir, f'{name}.json'), 'r') as f:
        args_dict = json.load(f)
    args = argparse.Namespace()
    for k,v in args_dict.items():setattr(args, k, v)
    return args
        
def save_params(params, name, save_dir):
    with open(os.path.join(save_dir,name), 'wb') as f: 
        pickle.dump(params, f)

def load_params(name, save_dir):
    with open(os.path.join(save_dir, name), 'rb') as f: 
        params = pickle.load(f)
    return params
        
def plot_training_curves(avg_elbos, experiments_folder, avg_evidence=None):


    num_fits = len(avg_elbos)
    fig, axes = plt.subplots(1, num_fits, figsize=(20,10))
    for fit_nb in range(num_fits):
        axes[fit_nb].set_yscale('symlog')
        axes[fit_nb].plot(avg_elbos[fit_nb], label='$\mathcal{L}(\\theta,\\phi)$')
        if avg_evidence is not None:
            axes[fit_nb].axhline(y=avg_evidence, c='red', label = '$log p_{\\theta}(x)$')
        axes[fit_nb].set_xlabel('Epoch') 
        axes[fit_nb].set_title(f'Fit {fit_nb+1}')
        axes[fit_nb].legend()
    plt.savefig(os.path.join(experiments_folder, 'training_curves'))
    plt.clf()


def plot_example_smoothed_states(p, q, theta, phi, state_seqs, obs_seqs, seq_nb, figname, *args):

    fig, (ax0, ax1) = plt.subplots(1,2, sharey=True, figsize=(20,10))
    plot_relative_errors_1D(ax0, state_seqs[seq_nb], *p.smooth_seq(obs_seqs[seq_nb], theta, *args))
    ax0.set_title('True params')

    plot_relative_errors_1D(ax1, state_seqs[seq_nb], *q.smooth_seq(obs_seqs[seq_nb], phi, *args))
    ax1.set_title('Fitted params')

    plt.tight_layout()
    plt.autoscale(True)
    plt.savefig(figname)
    plt.clf()



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

        ref_smoothed_means = ref_backwd_pass(tree_get_idx(-1, ref_filt_seq), ref_backwd_seq)[0]
        approx_smoothed_means = approx_backwd_pass(tree_get_idx(-1, approx_filt_seq), approx_backwd_seq)[0]
        vi_vs_kalman_marginals = jnp.abs(ref_smoothed_means - approx_smoothed_means)[jnp.array(timesteps)]

        return kalman_wrt_states, vi_wrt_states, vi_vs_kalman, vi_vs_kalman_marginals


    state_seqs, obs_seqs = vmap(ref_smoother.sample_seq, in_axes=(0,None,None))(random.split(key, 2), ref_params, seq_length)

    fig, (ax0, ax1, ax2, ax3) = plt.subplots(1,4, figsize=(20,10))

    
    for seq_nb, (state_seq, obs_seq) in tqdm(enumerate(zip(state_seqs, obs_seqs))):
        kalman_wrt_states, vi_wrt_states, vi_vs_kalman, vi_vs_kalman_marginals = results_for_single_seq(state_seq, obs_seq)
        ax0.plot(timesteps, kalman_wrt_states, label = f'Sequence {seq_nb}', linestyle='dotted', marker='.')
        ax1.plot(timesteps, vi_wrt_states, label = f'Sequence {seq_nb}', linestyle='dotted', marker='.')
        ax2.plot(timesteps, vi_vs_kalman, label = f'Sequence {seq_nb}', linestyle='dotted', marker='.')
        ax3.plot(timesteps, vi_vs_kalman_marginals, label = f'Sequence {seq_nb}', linestyle='dotted', marker='.')

    
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
    plt.autoscale(True)
    plt.tight_layout()


def multiple_length_ffbsi_smoothing(obs_seqs, smoother, params, timesteps, key, num_particles):
    
    key, subkey = random.split(key, 2)
    compute_filt_seq = jit(lambda obs_seq: smoother.compute_filt_seq(obs_seq, params, key, num_particles))
    backwd_pass = lambda filt_seq: smoother.backwd_pass(filt_seq, params, subkey)


    def results_for_single_seq(obs_seq):

        filt_seq = compute_filt_seq(obs_seq)

        results = []
        
        for length in tqdm(timesteps): 
            results.append(backwd_pass(tree_get_slice(0, length, filt_seq))[0])

        results.append(backwd_pass(filt_seq))


        return results

    results = []
    for obs_seq in tqdm(obs_seqs):
        results.append(results_for_single_seq(obs_seq))



    return results 

def multiple_length_linear_backward_smoothing(obs_seqs, smoother, params, timesteps):
    
    compute_filt_seq = lambda obs_seq: smoother.compute_filt_seq(obs_seq, params)
    compute_backwd_seq = lambda filt_seq: smoother.compute_backwd_seq(filt_seq, params)
    backwd_pass = smoother.backwd_pass

    def results_for_single_seq(obs_seq):

        filt_seq = compute_filt_seq(obs_seq)
        backwd_seq = compute_backwd_seq(filt_seq)

        results = []
        
        for length in tqdm(timesteps): 
        
            results.append(backwd_pass(tree_get_idx(length, filt_seq), tree_get_slice(0,length-1, backwd_seq))[0])

        results.append(backwd_pass(tree_get_idx(-1, filt_seq), backwd_seq))


        return results

    results = []
    for obs_seq in tqdm(obs_seqs):
        results.append(results_for_single_seq(obs_seq))



    return results 



def plot_multiple_length_smoothing(ref_state_seqs, ref_results, approx_results, timesteps, ref_name, approx_name, save_dir):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2,3, figsize=(20,15))
    xaxis = list(timesteps) + [len(ref_state_seqs[0])]




    if isinstance(approx_results, dict):
        for seq_nb, (ref_state_seq, ref_results_seq) in enumerate(zip(ref_state_seqs, ref_results)):
            ref_vs_states_additive =  []
            for i, length in enumerate(timesteps):
                ref_vs_states_additive.append(jnp.abs(jnp.sum(ref_results_seq[i] - ref_state_seq[:length], axis=0)))
            ref_vs_states_additive.append(jnp.abs(jnp.sum(ref_results_seq[-1][0] - ref_state_seq, axis=0)))
            ax0.plot(xaxis, ref_vs_states_additive, linestyle='dotted', marker='.', c='k')

        idx_color = 0
        handles = []
        for epoch_nb, approx_results_at_epoch in approx_results.items():
            c = colors[idx_color]
            for seq_nb, (ref_state_seq, ref_results_seq, approx_results_seq) in enumerate(zip(ref_state_seqs, ref_results, approx_results_at_epoch)):
                approx_vs_states_additive = []
                ref_vs_approx_additive = []

                for i, length in enumerate(timesteps):
                    approx_vs_states_additive.append(jnp.abs(jnp.sum(approx_results_seq[i] - ref_state_seq[:length], axis=0)))
                    ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[i] - ref_results_seq[i], axis=0)))

                approx_vs_states_additive.append(jnp.abs(jnp.sum(approx_results_seq[-1][0] - ref_state_seq, axis=0)))
                ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[-1][0] - ref_results_seq[-1][0], axis=0)))
                ref_vs_approx_marginals = jnp.abs(ref_results_seq[-1][0] - approx_results_seq[-1][0])

                ax1.plot(xaxis, approx_vs_states_additive, linestyle='dotted', marker='.', c=c, label=f'Epoch {epoch_nb}')
                ax2.plot(xaxis, ref_vs_approx_additive, linestyle='dotted', marker='.', c=c, label=f'Epoch {epoch_nb}')
                handle, = ax3.plot(ref_vs_approx_marginals, linestyle='dotted', marker='.', c=c, label=f'Epoch {epoch_nb}')
            handles.append(handle)
            idx_color+=1

        ax1.legend(handles=handles)
        ax2.legend(handles=handles)
        ax3.legend(handles=handles)

        ax0.set_title(f'{ref_name} vs states (additive)')
        ax0.set_xlabel('Sequence length')

        ax1.set_title(f'{approx_name} vs states (additive)')
        ax1.set_xlabel('Sequence length')

        ax2.set_title(f'{approx_name} vs {ref_name} (additive)')
        ax2.set_xlabel('Sequence length')

        ax3.set_title(f'{approx_name} vs {ref_name} (marginals)')
        ax3.set_xlabel('Sequence length')

        plot_relative_errors_1D(ax4, ref_state_seqs[0], *ref_results[0][-1])
        ax4.set_title(f'{ref_name} example smoothing')

        plot_relative_errors_1D(ax5, ref_state_seqs[0], *approx_results_at_epoch[0][-1])
        ax5.set_title(f'{approx_name} smoothing with fully fitted params')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'smoothing_results'))
        plt.clf()

    else: 

        for seq_nb, (ref_state_seq, ref_results_seq) in enumerate(zip(ref_state_seqs, ref_results)):
            ref_vs_states_additive =  []
            for i, length in enumerate(timesteps):
                ref_vs_states_additive.append(jnp.abs(jnp.sum(ref_results_seq[i] - ref_state_seq[:length], axis=0)))
            ref_vs_states_additive.append(jnp.abs(jnp.sum(ref_results_seq[-1][0] - ref_state_seq, axis=0)))
            ax0.plot(xaxis, ref_vs_states_additive, linestyle='dotted', marker='.', label=f'Seq {seq_nb}')


        for seq_nb, (ref_state_seq, ref_results_seq, approx_results_seq) in enumerate(zip(ref_state_seqs, ref_results, approx_results)):
            approx_vs_states_additive = []
            ref_vs_approx_additive = []

            for i, length in enumerate(timesteps):
                approx_vs_states_additive.append(jnp.abs(jnp.sum(approx_results_seq[i] - ref_state_seq[:length], axis=0)))
                ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[i] - ref_results_seq[i], axis=0)))

            approx_vs_states_additive.append(jnp.abs(jnp.sum(approx_results_seq[-1][0] - ref_state_seq, axis=0)))
            ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[-1][0] - ref_results_seq[-1][0], axis=0)))
            ref_vs_approx_marginals = jnp.abs(ref_results_seq[-1][0] - approx_results_seq[-1][0])

            ax1.plot(xaxis, approx_vs_states_additive, linestyle='dotted', marker='.', label=f'Seq {seq_nb}')
            ax2.plot(xaxis, ref_vs_approx_additive, linestyle='dotted', marker='.', label=f'Seq {seq_nb}')
            handle, = ax3.plot(ref_vs_approx_marginals, linestyle='dotted', marker='.',label=f'Seq {seq_nb}')

        ax0.legend()
        ax1.legend()
        ax2.legend()
        ax3.legend()

    

        ax0.set_title(f'{ref_name} vs states (additive)')
        ax0.set_xlabel('Sequence length')

        ax1.set_title(f'{approx_name} vs states (additive)')
        ax1.set_xlabel('Sequence length')

        ax2.set_title(f'{approx_name} vs {ref_name} (additive)')
        ax2.set_xlabel('Sequence length')

        ax3.set_title(f'{approx_name} vs {ref_name} (marginals)')
        ax3.set_xlabel('Sequence length')

        plot_relative_errors_1D(ax4, ref_state_seqs[0], *ref_results[0][-1])
        ax4.set_title(f'{ref_name} example smoothing')

        plot_relative_errors_1D(ax5, ref_state_seqs[0], *approx_results[0][-1])
        ax5.set_title(f'{approx_name} smoothing with fully fitted params')

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'smoothing_results'))
        plt.clf()

