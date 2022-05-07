from collections import namedtuple
from dataclasses import dataclass

from jax import numpy as jnp, vmap, config, random, jit, scipy as jsp
from functools import update_wrapper
from jax.tree_util import register_pytree_node_class, tree_map
from jax.scipy.linalg import solve_triangular
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import json
import os 
import pickle 
import argparse
from typing import Any
# Containers for parameters of various objects 

def enable_x64(use_x64=True):
    """
    Changes the default array type to use 64 bit precision as in NumPy.
    :param bool use_x64: when `True`, JAX arrays will use 64 bits by default;
        else 32 bits.
    """
    if not use_x64:
        use_x64 = os.getenv("JAX_ENABLE_X64", 0)
    config.update("jax_enable_x64", use_x64)


def set_platform(platform=None):
    """
    Changes platform to CPU, GPU, or TPU. This utility only takes
    effect at the beginning of your program.
    :param str platform: either 'cpu', 'gpu', or 'tpu'.
    """
    if platform is None:
        platform = os.getenv("JAX_PLATFORM_NAME", "cpu")
    config.update("jax_platform_name", platform)


def set_host_device_count(n):
    """
    By default, XLA considers all CPU cores as one device. This utility tells XLA
    that there are `n` host (CPU) devices available to use. As a consequence, this
    allows parallel mapping in JAX :func:`jax.pmap` to work in CPU platform.
    .. note:: This utility only takes effect at the beginning of your program.
        Under the hood, this sets the environment variable
        `XLA_FLAGS=--xla_force_host_platform_device_count=[num_devices]`, where
        `[num_device]` is the desired number of CPU devices `n`.
    .. warning:: Our understanding of the side effects of using the
        `xla_force_host_platform_device_count` flag in XLA is incomplete. If you
        observe some strange phenomenon when using this utility, please let us
        know through our issue or forum page. More information is available in this
        `JAX issue <https://github.com/google/jax/issues/1408>`_.
    :param int n: number of CPU devices to use.
    """
    xla_flags = os.getenv("XLA_FLAGS", "")
    xla_flags = re.sub(
        r"--xla_force_host_platform_device_count=\S+", "", xla_flags
    ).split()
    os.environ["XLA_FLAGS"] = " ".join(
        ["--xla_force_host_platform_device_count={}".format(n)] + xla_flags
    )



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


KernelParams = namedtuple('KernelParams', ['map','scale'])
BackwardState = namedtuple('BackwardState', ['shared', 'varying', 'inner'])


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

def empty_add(d):
    return jnp.zeros((d,d))

@register_pytree_node_class
class GaussianParams: 

    def __init__(self, mean=None, scale=None, eta1=None, eta2=None):

        if (mean is not None) and (scale is not None):
            self.mean = mean 
            self.scale = scale
        elif (eta1 is not None) and (eta2 is not None):
            self.eta1 = eta1 
            self.eta2 = eta2

    @classmethod
    def from_mean_scale(cls, mean, scale):
        obj = cls.__new__(cls)
        obj.mean = mean 
        obj.scale = scale
        return obj

    @classmethod
    def from_nat_params(cls, eta1, eta2):
        obj = cls.__new__(cls)
        obj.eta1 = eta1
        obj.eta2 = eta2 
        return obj

    @classmethod
    def from_vec(cls, vec, d, chol_add=empty_add):
        mean = vec[:d]
        chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(vec[d:])
        return cls(mean=mean, scale=Scale(chol=chol + chol_add(d)))
    
    @property
    def vec(self):
        d = self.mean.shape[0]
        return jnp.concatenate((self.mean, self.scale.chol[jnp.tril_indices(d)]))

    @lazy_property
    def mean(self):
        return self.scale.cov @ self.eta1

    @lazy_property
    def scale(self):
        return Scale(prec=-0.5*self.eta2)
    
    @lazy_property
    def eta1(self):
        return self.scale.prec @ self.mean 
        
    @lazy_property
    def eta2(self):
        return -0.5 * self.scale.prec 
        
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

# GaussianParams = namedtuple('GaussianParams', ['mean', 'scale'])

@register_pytree_node_class
class LinearMapParams:
    def __init__(self, w, b=None):
        self.w = w 
        if b is not None: 
            self.b = b
        
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

@register_pytree_node_class
@dataclass(init=True)
class HMMParams:
    
    prior: GaussianParams 
    transition: KernelParams
    emission: KernelParams

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

@register_pytree_node_class
@dataclass(init=True)
class NeuralBackwardSmootherParams:

    prior: GaussianParams 
    transition:KernelParams
    filt_update:KernelParams
    backwd_map:Any

    def compute_covs(self):
        self.prior.scale.cov
        self.transition.scale.cov

    def tree_flatten(self):
        return ((self.prior, self.transition, self.filt_update, self.backwd_map), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


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
        stored_epoch_nbs_for_fit = stored_epoch_nbs[fit_nb]
        plt.yscale('symlog')
        ydata = avg_elbos[fit_nb]
        plt.plot(range(len(ydata)), ydata, label='$\mathcal{L}(\\theta,\\phi)$', c='black')
        plt.axhline(y=avg_evidence, c='black', label = '$log p_{\\theta}(x)$', linestyle='dotted')
        idx_color = 0
        for epoch_nb in stored_epoch_nbs_for_fit:
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
    plt.plot(range(len(ydata)), ydata, label='$\mathcal{L}(\\theta,\\phi)$,'+f'{name1}', c=colors[0])
    ydata = avg_elbos_2[best_fit_idx_2]
    plt.plot(range(len(ydata)), ydata, label='$\mathcal{L}(\\theta,\\phi)$,'+f'{name2}', c=colors[1])
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

    compute_ref_filt_seq = lambda obs_seq: ref_smoother.compute_filt_state_seq(obs_seq, ref_params)
    compute_ref_backwd_seq = lambda filt_seq: ref_smoother.compute_kernel_state_seq(filt_seq, ref_params)

    compute_approx_filt_seq = lambda obs_seq: approx_smoother.compute_filt_state_seq(obs_seq, approx_params)
    compute_approx_backwd_seq = lambda filt_seq: approx_smoother.compute_kernel_state_seq(filt_seq, approx_params)
    
    ref_compute_marginals = ref_smoother.compute_marginals
    approx_compute_marginals = approx_smoother.compute_marginals

    def results_for_single_seq(state_seq, obs_seq):

        ref_filt_seq, approx_filt_seq = compute_ref_filt_seq(obs_seq), compute_approx_filt_seq(obs_seq)
        ref_backwd_seq, approx_backwd_seq = compute_ref_backwd_seq(ref_filt_seq), compute_approx_backwd_seq(approx_filt_seq)
        kalman_wrt_states, vi_wrt_states, vi_vs_kalman = [], [], []

        def result_up_to_length(length):

            ref_smoothed_means = ref_compute_marginals(tree_get_idx(length, ref_filt_seq), tree_get_slice(0,length-1, ref_backwd_seq))[0]
            approx_smoothed_means = approx_compute_marginals(tree_get_idx(length, approx_filt_seq), tree_get_slice(0,length-1, approx_backwd_seq))[0]
            
            kalman_wrt_states = jnp.abs(jnp.sum(ref_smoothed_means - state_seq[:length], axis=0))
            vi_wrt_states = jnp.abs(jnp.sum(approx_smoothed_means - state_seq[:length], axis=0))
            vi_vs_kalman = jnp.abs(jnp.sum(approx_smoothed_means - ref_smoothed_means, axis=0))

            return kalman_wrt_states, vi_wrt_states, vi_vs_kalman
        
        for length in timesteps: 
            result = result_up_to_length(length)
            kalman_wrt_states.append(result[0])
            vi_wrt_states.append(result[1])
            vi_vs_kalman.append(result[2])

        ref_smoothed_means = ref_compute_marginals(tree_get_idx(-1, ref_filt_seq), ref_backwd_seq)[0]
        approx_smoothed_means = approx_compute_marginals(tree_get_idx(-1, approx_filt_seq), approx_backwd_seq)[0]
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

def multiple_length_ffbsi_smoothing(key, obs_seqs, smoother, params, timesteps):
    
    params = smoother.format_params(params)
    key, subkey = random.split(key, 2)
    compute_filt_state_seq = jit(lambda obs_seq: smoother.compute_filt_state_seq(key, obs_seq, params))
    compute_marginals = lambda filt_seq: smoother.compute_marginals(subkey, filt_seq, params)


    def results_for_single_seq(obs_seq):

        filt_seq = compute_filt_state_seq(obs_seq)

        results = []
        
        for length in tqdm(timesteps): 
            results.append(compute_marginals(tree_get_slice(0, length, filt_seq))[0])

        results.append(compute_marginals(filt_seq))


        return results

    results = []
    for obs_seq in tqdm(obs_seqs):
        results.append(results_for_single_seq(obs_seq))



    return results 

def multiple_length_linear_backward_smoothing(obs_seqs, smoother, params, timesteps):
    
    params = smoother.format_params(params)
    compute_filt_state_seq = lambda obs_seq: smoother.compute_filt_state_seq(obs_seq, params)
    compute_kernel_state_seq = lambda filt_seq: smoother.compute_kernel_state_seq(filt_seq, params)
    compute_marginals = smoother.compute_marginals

    def results_for_single_seq(obs_seq):

        filt_seq = compute_filt_state_seq(obs_seq)
        backwd_seq = compute_kernel_state_seq(filt_seq)

        results = []
        
        for length in tqdm(timesteps): 
            marginal_means = compute_marginals(tree_get_idx(length, filt_seq), tree_get_slice(0,length-1, backwd_seq)).mean
            results.append(marginal_means)

        marginals = compute_marginals(tree_get_idx(-1, filt_seq), backwd_seq)
        results.append((marginals.mean, marginals.scale.cov))


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
