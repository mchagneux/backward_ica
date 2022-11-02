from dataclasses import dataclass

from jax import numpy as jnp, vmap, config, random, jit, scipy as jsp, lax
from functools import update_wrapper, partial
from jax.tree_util import register_pytree_node_class, tree_map
from jax.scipy.linalg import solve_triangular
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os 
import dill 
import argparse
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
from jaxlib.xla_extension import DeviceArray
# Containers for parameters of various objects 

# @partial(jit, static_argnums=(1,))
def moving_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: lax.dynamic_slice_in_dim(a, start, size))(starts)




_conditionnings = {'diagonal':lambda param, d: jnp.diag(param),
                'sym_def_pos': lambda param, d: mat_from_chol_vec(param, d) + jnp.eye(d),
                None:lambda x, d:x,
                'init_sym_def_pos': lambda x,d:x}


## config routines and model selection

def get_defaults(args):
    import math
    args.float64 = True

    args.default_prior_mean = 0.0 # default value for the mean of Gaussian prior
    args.default_prior_base_scale = math.sqrt(1e-2) # default value for the diagonal components of the covariance matrix of the prior
    args.default_transition_base_scale = math.sqrt(1e-2) # default value for the diagonal components of the covariance matrix of the transition kernel
    args.default_transition_bias = 0.0
    args.default_emission_base_scale = math.sqrt(1e-2)

    if 'chaotic_rnn' not in args.model:
        args.transition_matrix_conditionning = 'diagonal'
        if not(hasattr(args, 'transition_bias')):
            args.transition_bias = False
        args.range_transition_map_params = [0.9,0.99] # range of the components of the transition matrix

    else:
        args.range_transition_map_params = [-1,1] # range of the components of the transition matrix
        args.transition_matrix_conditionning = 'init_sym_def_pos' # constraint
        args.default_transition_matrix = os.path.join(args.load_from, 'W.npy')
        args.grid_size = 0.001 # discretization parameter for the chaotic rnn
        args.gamma = 2.5 # gamma for the chaotic rnn
        args.tau = 0.025 # tau for the chaotic rnn

        args.emission_matrix_conditionning = 'diagonal'
        args.range_emission_map_params = (0.99,1)
        args.default_emission_df = 2 # degrees of freedom for the emission noise
        args.default_emission_matrix = 1.0 # diagonal values for the emission matrix

    if not(hasattr(args, 'emission_bias')):
        args.emission_bias = False 

    if 'nonlinear_emission' in args.model:
        args.emission_map_layers = (8,)
        args.slope = 0 # amount of linearity in the emission function
        args.injective = True

    if 'neural_backward' in args.model:
        ## variational family
        args.update_layers = (8,8) # number of layers in the GRU which updates the variational filtering dist
        args.backwd_map_layers = (32,) # number of layers in the MLP which predicts backward parameters (not used in the Johnson method)

    elif 'johnson' in args.model:
        args.update_layers = (8,8)
        args.anisotropic = 'anisotropic' in args.model

    args.parametrization = 'cov_chol' # parametrization of the covariance matrices 


    args.num_particles = 10000 # number of particles for bootstrap filtering step
    args.num_smooth_particles = 1000 # number of particles for the FFBSi ancestral sampling step

    return args

def enable_x64(use_x64=True):
    """
    Changes the default array type to use 64 bit precision as in NumPy.
    :param bool use_x64: when `True`, JAX arrays will use 64 bits by default;
        else 32 bits.
    """
    if not use_x64:
        use_x64 = os.getenv("JAX_ENABLE_X64", 0)
    config.update("jax_enable_x64", use_x64)
    if use_x64: print('Using float64.')
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
    allows parkallel mapping in JAX :func:`jax.pmap` to work in CPU platform.
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


## misc. JAX indexing tools

def tree_get_strides(stride, tree):
    return tree_map(partial(moving_window, size=stride), tree)

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
    return tree_map(lambda a: lax.dynamic_slice_in_dim(a, start, stop-start), tree)


## quadratic forms and Gaussian subroutines 

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

def constant_terms_from_log_gaussian(dim:int, log_det:float)->float:
    """Utility function to compute the log of the term that is against the exponential for a multivariate Normal

    Args:
        dim (int): the dimension of the support of the multivariate Normal
        det (float): the precomputed determinant of the covariance matrix 

    Returns:
        float: the value of the requested factor  
    """

    return -0.5*(dim * jnp.log(2*jnp.pi) + log_det)

def transition_term_integrated_under_backward(q_backwd_params, transition_params):
    # expectation of the quadratic form that appears in the log of the state transition density

    A = transition_params.map.w @ q_backwd_params.map.w - jnp.eye(transition_params.noise.scale.cov.shape[0])
    b = transition_params.map.w @ q_backwd_params.map.b + transition_params.map.b
    Omega = transition_params.noise.scale.prec
    
    result = -0.5 * QuadTerm.from_A_b_Omega(A, b, Omega)
    result.c += -0.5 * jnp.trace(transition_params.noise.scale.prec @ transition_params.map.w @ q_backwd_params.noise.scale.cov @ transition_params.map.w.T) \
                + constant_terms_from_log_gaussian(transition_params.noise.scale.cov.shape[0], transition_params.noise.scale.log_det)
    return result 

def expect_quadratic_term_under_backward(quad_form:QuadTerm, backwd_params):
    # the result is still a quadratic forms with new parameters, following the formula for expected values of quadratic forms  

    W = backwd_params.map.w.T @ quad_form.W @ backwd_params.map.w
    v = backwd_params.map.w.T @ (quad_form.v + (quad_form.W + quad_form.W.T) @ backwd_params.map.b)
    c = quad_form.c + jnp.trace(quad_form.W @ backwd_params.noise.scale.cov) \
                    + backwd_params.map.b.T @ quad_form.W @ backwd_params.map.b  \
                    + quad_form.v.T @ backwd_params.map.b 

    return QuadTerm(W=W, v=v, c=c)

def expect_quadratic_term_under_gaussian(quad_form:QuadTerm, gaussian_params):
    return jnp.trace(quad_form.W @ gaussian_params.scale.cov) + quad_form.evaluate(gaussian_params.mean)

def quadratic_term_from_log_gaussian(gaussian_params):

    result = - 0.5 * QuadTerm(W=gaussian_params.scale.prec, 
                    v=-(gaussian_params.scale.prec + gaussian_params.scale.prec.T) @ gaussian_params.mean, 
                    c=gaussian_params.mean.T @ gaussian_params.scale.prec @ gaussian_params.mean)

    result.c += constant_terms_from_log_gaussian(gaussian_params.mean.shape[0], gaussian_params.scale.log_det)

    return result

def get_tractable_emission_term(obs, emission_params):
    A = emission_params.map.w
    b = emission_params.map.b - obs
    Omega = emission_params.noise.scale.prec
    emission_term = -0.5*QuadTerm.from_A_b_Omega(A, b, Omega)
    emission_term.c += constant_terms_from_log_gaussian(emission_params.noise.scale.cov.shape[0], emission_params.noise.scale.log_det)
    return emission_term

def get_tractable_emission_term_from_natparams(emission_natparams):
    eta1, eta2 = emission_natparams
    const = -0.25 * eta1.T @ jnp.linalg.solve(eta2, eta1) - 0.5 * jnp.log(jnp.linalg.det(-2*eta2)) - eta1.shape[0] * jnp.log(jnp.pi)
    return QuadTerm(W=eta2, 
                    v=eta1, 
                    c=const)


## covariance matrices tools 

def chol_from_vec(vec, d):

    return jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(vec)

def mat_from_chol_vec(vec, d):
    w = chol_from_vec(vec,d)
    return w @ w.T

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

def mat_from_chol(chol):
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



## tool for lazy eval 
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



## normalizers 
def exp_and_normalize(x):

    x = jnp.exp(x - x.max())
    return x / x.sum()




def params_to_dict(params):
    if isinstance(params, np.ndarray) or isinstance(params, DeviceArray):
        return params
    elif isinstance(params, dict):
        for key, value in params.items():
            params[key] = params_to_dict(value)
        return params
    elif hasattr(params, '__dict__'):
        return params_to_dict(vars(params))
    elif hasattr(params, '_asdict'): 
        return params_to_dict(params._asdict())
    else:
        return params_to_dict({k:v for k,v in enumerate(params)})

def params_to_flattened_dict(params):
    params_dict = params_to_dict(params)
    return pd.json_normalize(params_dict, sep='/').to_dict(orient='records')[0]
    
def empty_add(d):
    return jnp.zeros((d,d))




def plot_relative_errors_1D(ax, pred_means, pred_covs, color='black', alpha=0.2, hatch=None, label=''):
    # up_to = 64
    pred_means, pred_covs = pred_means.squeeze(), pred_covs.squeeze()
    time_axis = range(len(pred_means))
    yerr = 1.96 * jnp.sqrt(pred_covs)
    upper = pred_means + yerr 
    lower = pred_means - yerr 

    ax.plot(time_axis, pred_means, linestyle='dashed', c=color, label=label)
    ax.fill_between(time_axis, lower, upper, alpha=alpha, color=color, hatch=hatch)

def plot_relative_errors_2D(ax, true_sequence, pred_means, pred_covs, limit=False):
    # up_to = 64
    true_sequence, pred_means, pred_covs = true_sequence.squeeze(), pred_means.squeeze(), pred_covs.squeeze()
    if limit: true_sequence, pred_means, pred_covs = true_sequence[:limit], pred_means[:limit], pred_covs[:limit]
    # time_axis = range(len(true_sequence))
    ax.scatter(true_sequence[:,0], true_sequence[:,1], c='r')
    for mean, cov in zip(pred_means, pred_covs):
        confidence_ellipse(mean, cov, ax, c='b')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()


## serializations 

def save_args(args, name, save_dir):
    with open(os.path.join(save_dir, f'{name}.json'), 'w') as f:
        args_dict = vars(args)
        json.dump(args_dict, f, indent=4)

def load_args(name, save_dir):
    with open(os.path.join(save_dir, f'{name}.json'), 'r') as f:
        args_dict = json.load(f)
    args = argparse.Namespace()
    for k,v in args_dict.items():setattr(args, k, v)
    return args
        
def save_params(params, name, save_dir):
    with open(os.path.join(save_dir,name), 'wb') as f: 
        dill.dump(params, f)

def load_params(name, save_dir):
    with open(os.path.join(save_dir, name), 'rb') as f: 
        params = dill.load(f)
    return params
        
def load_smoothing_results(save_dir):
    with open(os.path.join(save_dir, 'smoothing_results'), 'rb') as f: 
        results = dill.load(f)
    return results

def save_train_logs(train_logs, save_dir, plot=True, best_epochs_only=False):
    with open(os.path.join(save_dir, 'train_logs'), 'wb') as f: 
        dill.dump(train_logs, f)
    if plot: 
        plot_training_curves(*train_logs, save_dir, plot_only=None, best_epochs_only=best_epochs_only)

def load_train_logs(save_dir):
    with open(os.path.join(save_dir, 'train_logs'), 'rb') as f: 
        train_logs = dill.load(f)
    return train_logs
        

# def plot_to_image(figure):
#     """Converts the matplotlib plot specified by 'figure' to a PNG image and
#     returns it. The supplied figure is closed and inaccessible after this call."""
#     # Save the plot to a PNG in memory.
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     # Closing the figure prevents it from being displayed directly inside
#     # the notebook.
#     plt.close(figure)
#     buf.seek(0)
#     # Convert PNG buffer to TF image
#     image = tf.image.decode_png(buf.getvalue(), channels=4)
#     # Add the batch dimension
#     image = tf.expand_dims(image, 0)
#     return image



## OLD 
def plot_training_curves(best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence, save_dir, plot_only, best_epochs_only):
    plt.rcParams.update({'font.size': 10.35})

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    num_fits = len(avg_elbos)
    for fit_nb in range(num_fits):

        if best_epochs_only: stored_epoch_nbs_for_fit = [stored_epoch_nbs[fit_nb]]
        else: stored_epoch_nbs_for_fit = stored_epoch_nbs[0][fit_nb]
        ydata = avg_elbos[fit_nb][1:]
        plt.plot(range(len(ydata)), ydata, label='$\mathcal{L}(\\theta,\\phi)$', c='black')
        plt.axhline(y=avg_evidence, c='black', label = '$log p_{\\theta}(x)$', linestyle='dotted')
        idx_color = 0
        for epoch_nb in stored_epoch_nbs_for_fit:
            if plot_only is not None:
                if epoch_nb in plot_only:
                    plt.axvline(x=epoch_nb, linestyle='dashed', c=colors[idx_color])
                    idx_color+=1
            else: 
                plt.axvline(x=epoch_nb, linestyle='dashed', c=colors[idx_color])
                idx_color+=1               

        plt.xlabel('Epoch') 
        # plt.legend()
        
        if fit_nb == best_fit_idx: plt.savefig(os.path.join(save_dir, f'training_curve_fit_{fit_nb}(best).pdf'), format='pdf')
        else: plt.savefig(os.path.join(save_dir, f'training_curve_fit_{fit_nb}.pdf'), format='pdf')
        plt.clf()

def superpose_training_curves(avg_evidence, avg_elbos_list, method_names, save_dir, start_index=0):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for nb, (avg_elbos, method_name) in enumerate(zip(avg_elbos_list, method_names)):
        plt.plot(range(start_index, len(avg_elbos)+start_index), avg_elbos, label=f'{method_name}', c=colors[nb])
    plt.axhline(y=avg_evidence, c='black', label = '$log p_{\\theta}(x)$', linestyle='dotted')
    plt.xlabel('Epoch') 
    plt.legend()
    
    plt.show()
    plt.savefig(os.path.join(save_dir, 'comparison_of_training_curves.pdf'), format='pdf')
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

    compute_ref_filt_seq = lambda obs_seq: ref_smoother.compute_filt_params_seq(obs_seq, ref_params)
    compute_ref_backwd_seq = lambda filt_seq: ref_smoother.compute_backwd_params_seq(filt_seq, ref_params)

    compute_approx_filt_seq = lambda obs_seq: approx_smoother.compute_filt_params_seq(obs_seq, approx_params)
    compute_approx_backwd_seq = lambda filt_seq: approx_smoother.compute_backwd_params_seq(filt_seq, approx_params)
    
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
    params.compute_covs()
    compute_filt_params_seq = jit(lambda key, obs_seq: smoother.compute_filt_params_seq(key, obs_seq, params))
    compute_marginals = lambda key, filt_seq: smoother.compute_marginals(key, filt_seq, params)


    def results_for_single_seq(key_filt, key_back, obs_seq):

        filt_seq = compute_filt_params_seq(key_filt, obs_seq)

        results = []
        
        for length in tqdm(timesteps): 
            paths = compute_marginals(key_back, tree_get_slice(0, length, filt_seq))
            results.append(jnp.mean(paths, axis=0))

        paths = compute_marginals(key_back, filt_seq)
        results.append((jnp.mean(paths, axis=0), jnp.var(paths, axis=0)))


        return results

    results = []
    for obs_seq in tqdm(obs_seqs):
        key, key_filt, key_back = random.split(key, 3)

        results.append(results_for_single_seq(key_filt, key_back, obs_seq))



    return results 

def multiple_length_linear_backward_smoothing(obs_seqs, smoother, params, timesteps):
    
    params = smoother.format_params(params)
    compute_filt_params_seq = lambda obs_seq: smoother.compute_filt_params_seq(obs_seq, params)
    compute_backwd_params_seq = lambda filt_seq: smoother.compute_backwd_params_seq(filt_seq, params)
    compute_marginals = smoother.compute_marginals

    def results_for_single_seq(obs_seq):

        filt_seq = compute_filt_params_seq(obs_seq)
        backwd_seq = compute_backwd_params_seq(filt_seq)

        results = []
        
        for length in tqdm(timesteps): 
            marginal_means = compute_marginals(tree_get_idx(length-1, filt_seq), tree_get_slice(0, length-1, backwd_seq)).mean
            results.append(marginal_means)

        marginals = compute_marginals(tree_get_idx(-1, filt_seq), backwd_seq)
        results.append((marginals.mean, marginals.scale.cov))


        return results

    results = []
    for obs_seq in tqdm(obs_seqs):
        results.append(results_for_single_seq(obs_seq))



    return results 

def plot_multiple_length_smoothing(ref_state_seqs, ref_results, approx_results, timesteps, ref_name, approx_name, save_dir):

    plt.rcParams.update({'font.size': 10.35})
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    fig, ((ax0, ax1, ax2), (ax3, ax4, ax5)) = plt.subplots(2,3, figsize=(20,15))
    xaxis = list(timesteps) + [len(ref_state_seqs[0])]

    
    for seq_nb, (ref_state_seq, ref_results_seq) in enumerate(zip(ref_state_seqs, ref_results)):
        ref_vs_states_additive =  []
        for i, length in enumerate(timesteps):
            ref_vs_states_additive.append(jnp.abs(jnp.sum(ref_results_seq[i] - ref_state_seq[:length], axis=0)))
        ref_vs_states_additive.append(jnp.abs(jnp.sum(ref_results_seq[-1][0] - ref_state_seq, axis=0)))

        ax0.plot(xaxis, ref_vs_states_additive, linestyle='dotted', marker='.', c='k')

    handles = []
    for idx, (name, approx_results_of_method) in enumerate(approx_results.items()):
        c = colors[idx]
        for seq_nb, (ref_state_seq, ref_results_seq, approx_results_seq) in enumerate(zip(ref_state_seqs, ref_results, approx_results_of_method)):
            approx_vs_states_additive = []
            ref_vs_approx_additive = []

            for i, length in enumerate(timesteps):
                approx_vs_states_additive.append(jnp.abs(jnp.sum(approx_results_seq[i] - ref_state_seq[:length], axis=0)))
                ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[i] - ref_results_seq[i], axis=0)))

            approx_vs_states_additive.append(jnp.abs(jnp.sum(approx_results_seq[-1][0] - ref_state_seq, axis=0)))
            ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[-1][0] - ref_results_seq[-1][0], axis=0)))
            ref_vs_approx_marginals = jnp.abs(ref_results_seq[-1][0] - approx_results_seq[-1][0])

            ax1.plot(xaxis, approx_vs_states_additive, linestyle='dotted', marker='.', c=c, label=f'{name}')
            ax2.plot(xaxis, ref_vs_approx_additive, linestyle='dotted', marker='.', c=c, label=f'{name}')
            handle, = ax3.plot(ref_vs_approx_marginals, linestyle='dotted', marker='.', c=c, label=f'{name}')
        handles.append(handle)

    # ax0.set_title(f'{ref_name} vs states (additive)')
    ax0.set_xlabel('$n$')

    # ax1.set_title(f'{approx_name} vs states (additive)')
    ax1.set_xlabel('$n$')

    # ax2.set_title(f'{approx_name} vs {ref_name} (additive)')
    ax2.set_xlabel('$n$')

    # ax3.set_title(f'{approx_name} vs {ref_name} (marginals)')
    ax3.set_xlabel('$n$')

    plot_relative_errors_1D(ax4, ref_state_seqs[0], *ref_results[0][-1])
    ax4.set_title(f'{ref_name} smoothing')

    plot_relative_errors_1D(ax5, ref_state_seqs[0], *approx_results_of_method[0][-1])
    ax5.set_title(f'{name} smoothing')

    ax4.legend()
    ax5.legend()


    extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_dir,'smoothing_theta_star_vs_states.pdf'), bbox_inches=extent.expanded(2, 2), format='pdf')

    # Save just the portion _inside_ the second axis's boundaries
    extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_dir,'additive.pdf'), bbox_inches=extent.expanded(1.261, 1.2), format='pdf')

    extent = ax3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_dir,'marginal.pdf'), bbox_inches=extent.expanded(1.261, 1.2), format='pdf')

    extent = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_dir,'smoothing_theta_star.pdf'), bbox_inches=extent.expanded(1.261, 1.2), format='pdf')

    extent = ax5.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_dir,'smoothing_best_phi.pdf'), bbox_inches=extent.expanded(1.261, 1.2), format='pdf')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'smoothing_results.pdf'), format='pdf')
    plt.clf()

def compare_multiple_length_smoothing(ref_dir, eval_dirs, train_dirs, pretty_names, save_dir):
    
    plt.rcParams.update({'font.size': 20})

    
    train_logs = [load_train_logs(train_dir) for train_dir in train_dirs]
    
    avg_evidence = train_logs[0][-1]
    avg_elbos_list = [train_log[2][train_log[0]] for train_log in train_logs]

    superpose_training_curves(avg_evidence, avg_elbos_list, pretty_names, save_dir, start_index=0)


    colors = ['g','b','r', 'c', 'm', 'y', 'k']
    markers = ['*', '.', 'x','-']

    eval_args = load_args('eval_args', eval_dirs[0])
    key = random.PRNGKey(eval_args.seed)

    train_args = load_args('train_args', train_dirs[0])

    set_defaults(train_args)

    p = hmm.NonLinearHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                            range_transition_map_params=train_args.range_transition_map_params,
                            layers=train_args.emission_map_layers,
                            slope=train_args.slope,
                            transition_bias=train_args.transition_bias,
                            injective=train_args.injective) # specify the structure of the true model

    theta_star = load_params('theta', train_args.save_dir)
    key_gen, _ = random.split(key,2)
    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, eval_args.num_seqs, eval_args.seq_length)

    timesteps = range(2, eval_args.seq_length, eval_args.step)
    print(obs_seqs[0][list(timesteps)[0]])

    ref_results = load_smoothing_results(ref_dir)
    approx_results = {pretty_name:load_smoothing_results(eval_dir) for pretty_name, eval_dir in zip(pretty_names, eval_dirs)}
    fig, (ax0, ax1, ax2) = plt.subplots(3, figsize=(20,20))
    xaxis = list(timesteps) + [len(state_seqs[0])]
    mse = dict()
    mse['ffbsi'] = []
    for seq_nb, (ref_state_seq, ref_results_seq) in enumerate(zip(state_seqs, ref_results)):
        ref_vs_states_additive =  []
        for i, length in enumerate(timesteps):
            ref_vs_states_additive.append(jnp.abs(jnp.sum(ref_results_seq[i] - ref_state_seq[:length], axis=0)))
        ref_vs_states_additive.append(jnp.abs(jnp.sum(ref_results_seq[-1][0] - ref_state_seq, axis=0)))

        marginal_errors = ref_results_seq[-1][0] - ref_state_seq
        mse_seq = marginal_errors.flatten()**2
        mse['ffbsi'].append((jnp.mean(mse_seq).tolist(),jnp.var(mse_seq).tolist()))
        ax0.plot(xaxis, ref_vs_states_additive, linestyle='dotted', marker='.', c='k')

    handles = []
    for idx, (method_name, approx_results_for_method) in enumerate(approx_results.items()):
        mse[method_name] = []
        c = colors[idx]
        m = markers[idx]
        for seq_nb, (ref_state_seq, ref_results_seq, approx_results_seq) in enumerate(zip(state_seqs, ref_results, approx_results_for_method)):
            ref_vs_approx_additive = []

            for i, length in enumerate(timesteps):
                ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[i], axis=0) - jnp.sum(ref_results_seq[i], axis=0)))
                # ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[i] - ref_state_seq[:length], axis=0)))

            ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[-1][0], axis=0) - jnp.sum(ref_results_seq[-1][0], axis=0)))
            # ref_vs_approx_additive.append(jnp.abs(jnp.sum(approx_results_seq[-1][0] - ref_state_seq, axis=0)))

            marginals = approx_results_seq[-1][0] - ref_state_seq
            # ref_vs_approx_marginals = jnp.abs(ref_state_seq - approx_results_seq[-1][0])
            mse_seq = marginals.flatten()**2
            mse[method_name].append(ref_vs_approx_additive[-1].tolist())
            handle, = ax1.plot(xaxis, ref_vs_approx_additive, linestyle='dotted', marker=m, c=c, label=f'{method_name}')
            ax2.scatter(range(len(marginals)), mse_seq, c=c, label=f'{method_name}', marker=m)

        handles.append(handle)

    # ax1.legend(handles=handles)
    # ax2.legend(handles=handles)

    # ax0.set_title(f'FFBSi gt vs states (additive)')
    ax0.set_xlabel('$n$')

    # ax1.set_title(f'Learnt models vs ground truth smoothing (additive)')
    ax1.set_xlabel('$n$')

    # ax2.set_title(f'Learnt models vs ground truth smoothing (marginals)')
    ax2.set_xlabel('$n$')



    # extent = ax0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig(os.path.join(save_dir,'smoothing_theta_star_vs_states.pdf'), bbox_inches=extent.expanded(2, 2), format='pdf')

    # Save just the portion _inside_ the second axis's boundaries
    extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_dir,'additive.pdf'), bbox_inches=extent.expanded(1.261, 1.3), format='pdf')

    extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(save_dir,'marginal.pdf'), bbox_inches=extent.expanded(1.261, 1.3), format='pdf')

    # extent = ax4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig(os.path.join(save_dir,'smoothing_theta_star.pdf'), bbox_inches=extent.expanded(1.261, 1.2), format='pdf')

    # extent = ax5.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig(os.path.join(save_dir,'smoothing_best_phi.pdf'), bbox_inches=extent.expanded(1.261, 1.2), format='pdf')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'smoothing_results.pdf'), format='pdf')
    plt.clf()

    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5,1, figsize=(20,20))
    limit = 100
    

    # plot_relative_errors_1D(ax0, state_seqs[0], *ref_results[0][-1], limit)
    # ax0.set_title(f'FFBSi')

    # plot_relative_errors_1D(ax1, state_seqs[0], *approx_results[pretty_names[0]][0][-1], limit)
    # ax1.set_title(f'{pretty_names[0]}')

    # plot_relative_errors_1D(ax2, state_seqs[0], *approx_results[pretty_names[1]][0][-1], limit)
    # ax2.set_title(f'{pretty_names[1]}')

    # plot_relative_errors_1D(ax3, state_seqs[0], *approx_results[pretty_names[2]][0][-1], limit)
    # ax3.set_title(f'{pretty_names[2]}')
    with open(os.path.join(save_dir, 'mse_values.json'), 'w') as f:
        json.dump(mse, f, indent=4)
    # plot_relative_errors_1D(ax4, state_seqs[0], *approx_results['ffbsi_em_2022_05_18__17_59_59'][0][-1])
    # ax4.set_title(f'FFBSi EM')

    plt.savefig(os.path.join(save_dir, 'smoothing_visualizations.pdf'), format='pdf')

def confidence_ellipse(mean, cov, ax, c, n_std=1.96):


    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2, facecolor=c)
    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]
    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    ax.scatter(mean_x, mean_y, color=c)
    return ax.add_patch(ellipse)