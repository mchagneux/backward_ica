from collections import namedtuple
from dataclasses import dataclass
from jax import disable_jit, numpy as jnp, vmap, config, random, lax, jit, scipy as jsp
from functools import partial, update_wrapper
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

def tree_prepend(prep, tree):
    preprended = tree_map(
        lambda a, b: jnp.concatenate((a[None,:], b)), prep, tree
    )
    return preprended

def tree_append(tree, app):
    appended = tree_map(
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




def chol_from_prec(prec):
    # This formulation only takes the inverse of a triangular matrix
    # which is more numerically stable.
    # Refer to:
    # https://nbviewer.jupyter.org/gist/fehiepsi/5ef8e09e61604f10607380467eb82006#Precision-to-scale_tril
    tril_inv = jnp.swapaxes(
        jnp.linalg.cholesky(prec[..., ::-1, ::-1])[..., ::-1, ::-1], -2, -1
    )
    identity = jnp.broadcast_to(jnp.identity(prec.shape[-1]), tril_inv.shape)
    return jsp.linalg.solve_triangular(tril_inv, identity, lower=True)

def cov_from_chol(chol):
    return jnp.matmul(chol, jnp.swapaxes(chol, -1, -2))

def cholesky(mat):
    return jnp.linalg.cholesky(mat)

def inv_from_chol(chol):

    identity = jnp.broadcast_to(
        jnp.eye(chol.shape[-1]), chol.shape)

    return jsp.linalg.cho_solve((chol, True), identity)

def log_det_from_cov(cov):
    return log_det_from_chol(cholesky(cov))

def log_det_from_chol(chol):
    return jnp.sum(jnp.log(jnp.diagonal(chol)**2))

def inv(mat):
    return inv_from_chol(cholesky(mat))

def inv_of_chol(mat):
    return inv_of_chol_from_chol(cholesky(mat))

def inv_of_chol_from_chol(mat_chol):
    return solve_triangular(a=mat_chol, b=jnp.eye(mat_chol.shape[0]), lower=True)

# def scale_params_from_chol(chol):
#     return chol, cov_from_chol(chol), prec_from_chol(chol), log_det_from_chol(chol)

class lazy_property(object):
    r"""
    Used as a decorator for lazy loading of class attributes. This uses a
    non-data descriptor that calls the wrapped method to compute the property on
    first call; thereafter replacing the wrapped method into an instance
    attribute.
    """

    def __init__(self, wrapped):
        self.wrapped = wrapped
        update_wrapper(self, wrapped)

    # This is to prevent warnings from sphinx
    def __call__(self, *args, **kwargs):
        return self.wrapped(*args, **kwargs)

    def __get__(self, instance, obj_type=None):
        if instance is None:
            return self
        value = self.wrapped(instance)
        setattr(instance, self.wrapped.__name__, value)
        return value

@register_pytree_node_class
class Scale:

    def __init__(self, chol=None, cov=None, prec=None):

        if cov is not None:
            self.cov = cov
            self.chol = cholesky(cov)

        elif prec is not None:
            self.prec = prec
            self.chol = chol_from_prec(prec)

        elif chol is not None:
            self.chol = chol
            
        else:
            raise ValueError()        

    @lazy_property
    def cov(self):
        return cov_from_chol(self.chol)

    @lazy_property
    def prec(self):
        return inv_from_chol(self.chol)

    @lazy_property
    def log_det(self):
        return log_det_from_chol(self.chol)

    def tree_flatten(self):
        attrs = vars(self)
        children = attrs.values()
        aux_data = attrs.keys()
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, params):
        obj = cls.__new__(cls)
        for k,v in zip(aux_data, params):
            setattr(obj, k, v)
        return obj

    def __repr__(self):
        return str(vars(self))


KernelParams = namedtuple('KernelParams', ['map','scale'])
LinearMapParams = namedtuple('LinearMapParams', ['w', 'b'])
GaussianParams = namedtuple('GaussianParams', ['mean', 'scale'])

@register_pytree_node_class
@dataclass(init=True)
class HMMParams:
    
    prior: GaussianParams 
    transition:KernelParams
    emission:KernelParams

    def compute_covs(self):
        self.prior.scale.cov
        self.transition.scale.cov
        self.emission.scale.cov

    def tree_flatten(self):
        return ((self.prior, self.transition, self.emission), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@register_pytree_node_class
@dataclass(init=True)
class NeuralLinearBackwardSmootherParams:

    prior: GaussianParams 
    transition:KernelParams
    filt_update:KernelParams

    def compute_covs(self):
        self.prior.scale.cov
        self.transition.scale.cov

    def tree_flatten(self):
        return ((self.prior, self.transition, self.filt_update), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


# NeuralBackwardSmootherParams = namedtuple('NeuralLinearBackwardSmootherParams', ['prior','transition', 'filt_update'])



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
    ax.legend()

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
        
def save_train_logs(train_logs, save_dir, plot=True):
    with open(os.path.join(save_dir, 'train_logs'), 'wb') as f: 
        pickle.dump(train_logs, f)
    if plot: 
        plot_training_curves(*train_logs, save_dir)

def load_train_logs(save_dir):
    with open(os.path.join(save_dir, 'train_logs'), 'rb') as f: 
        train_logs = pickle.load(f)
    return train_logs
        
def plot_training_curves(best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence, save_dir):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    num_fits = len(avg_elbos)
    for fit_nb in range(num_fits):
        plt.yscale('symlog')
        ydata = avg_elbos[fit_nb]
        plt.plot(range(1,len(ydata)), ydata[1:], label='$\mathcal{L}(\\theta,\\phi)$', c='black')
        plt.axhline(y=avg_evidence, c='black', label = '$log p_{\\theta}(x)$', linestyle='dotted')
        idx_color = 0
        for epoch_nb in stored_epoch_nbs[len(stored_epoch_nbs)-3:]:
            plt.axvline(x=epoch_nb, linestyle='dashed', c=colors[idx_color])
            idx_color+=1

        plt.xlabel('Epoch') 
        plt.legend()
        
        if fit_nb == best_fit_idx: plt.savefig(os.path.join(save_dir, f'training_curve_fit_{fit_nb}(best)'))
        else: plt.savefig(os.path.join(save_dir, f'training_curve_fit_{fit_nb}'))
        plt.clf()

def superpose_training_curves(train_logs_1, train_logs_2, name1, name2, save_dir):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    best_fit_idx_1, _ , avg_elbos_1, avg_evidence = train_logs_1
    best_fit_idx_2, _ , avg_elbos_2, _ = train_logs_2

    plt.yscale('symlog')
    
    ydata = avg_elbos_1[best_fit_idx_1]
    plt.plot(range(1,len(ydata)), ydata[1:], label='$\mathcal{L}(\\theta,\\phi)$,'+f'{name1}', c=colors[0])
    ydata = avg_elbos_2[best_fit_idx_2]
    plt.plot(range(1,len(ydata)), ydata[1:], label='$\mathcal{L}(\\theta,\\phi)$,'+f'{name2}', c=colors[1])
    plt.axhline(y=avg_evidence, c='black', label = '$log p_{\\theta}(x)$', linestyle='dotted')

    plt.xlabel('Epoch') 
    plt.legend()
    
    plt.savefig(os.path.join(save_dir, 'comparison_of_training_curves'))
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

        handles = []
        for idx, (epoch_nb, approx_results_at_epoch) in enumerate(approx_results.items()):
            c = colors[idx]
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

def compare_multiple_length_smoothing(ref_state_seqs, ref_results, approx_results, timesteps, ref_name, approx_name, save_dir):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    fig, ((ax0, ax1, ax2, ax3), (ax4, ax5, ax6, ax7)) = plt.subplots(2,4, figsize=(20,15))
    xaxis = list(timesteps) + [len(ref_state_seqs[0])]

    ax7.set_axis_off()

    for seq_nb, (ref_state_seq, ref_results_seq) in enumerate(zip(ref_state_seqs, ref_results)):
        ref_vs_states_additive =  []
        for i, length in enumerate(timesteps):
            ref_vs_states_additive.append(jnp.abs(jnp.sum(ref_results_seq[i] - ref_state_seq[:length], axis=0)))
        ref_vs_states_additive.append(jnp.abs(jnp.sum(ref_results_seq[-1][0] - ref_state_seq, axis=0)))
        ax0.plot(xaxis, ref_vs_states_additive, linestyle='dotted', marker='.', c='k')

        ax4.set_title(f'{ref_name} example smoothing')
    plot_relative_errors_1D(ax4, ref_state_seqs[0], *ref_results[0][-1])

    handles = []
    for idx, (method_name, approx_results_for_method) in enumerate(approx_results.items()):
        c = colors[idx]
        for seq_nb, (ref_state_seq, ref_results_seq, approx_results_seq) in enumerate(zip(ref_state_seqs, ref_results, approx_results_for_method)):
            approx_vs_states_additive = []
            ref_vs_approx_additive = []

            for i, length in enumerate(timesteps):
                approx_vs_states_additive.append(jnp.abs(jnp.sum(approx_results_seq[i] - ref_state_seq[:length], axis=0)))
                ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[i] - ref_results_seq[i], axis=0)))

            approx_vs_states_additive.append(jnp.abs(jnp.sum(approx_results_seq[-1][0] - ref_state_seq, axis=0)))
            ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[-1][0] - ref_results_seq[-1][0], axis=0)))
            ref_vs_approx_marginals = jnp.abs(ref_results_seq[-1][0] - approx_results_seq[-1][0])

            ax1.plot(xaxis, approx_vs_states_additive, linestyle='dotted', marker='.', c=c, label=f'{method_name}')
            ax2.plot(xaxis, ref_vs_approx_additive, linestyle='dotted', marker='.', c=c, label=f'{method_name}')
            handle, = ax3.plot(ref_vs_approx_marginals, linestyle='dotted', marker='.', c=c, label=f'{method_name}')
        handles.append(handle)

    plot_relative_errors_1D(ax5, ref_state_seqs[0], *approx_results['linearVI'][0][-1])
    ax5.set_title('LinearVI smoothing with fully fitted params')
    
    plot_relative_errors_1D(ax6, ref_state_seqs[0], *approx_results['nonlinearVI'][0][-1])
    ax6.set_title('NonlinearVI smoothing with fully fitted params')


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


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'smoothing_results'))
    plt.clf()
