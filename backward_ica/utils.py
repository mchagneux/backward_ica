from dataclasses import dataclass

from jax import numpy as jnp, vmap, config, random, jit, scipy as jsp, lax, tree_util
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
import jax
from jaxlib.xla_extension import DeviceArray
import collections.abc

# Containers for parameters of various objects 


# @partial(jit, static_argnums=(1,))
def moving_window(a, size: int):
    starts = jnp.arange(len(a) - size + 1)
    return vmap(lambda start: lax.dynamic_slice_in_dim(a, start, size))(starts)



_conditionnings = {'diagonal':lambda param, d: jnp.diag(param),
                'sym_def_pos': lambda param, d: mat_from_chol_vec(param, d) + jnp.eye(d),
                None:lambda x, d:x,
                'init_sym_def_pos': lambda x,d:x}


def get_keys(key, num_seqs, num_epochs):
    keys = jax.random.split(key, num_seqs * num_epochs)
    keys = jnp.array(keys).reshape(num_epochs, num_seqs,-1)
    return keys

def get_dummy_keys(key, num_seqs, num_epochs): 
    return jnp.empty((num_epochs, num_seqs, 1))


## config routines and model selection


def elbo_h_0_online(data, models):

    p = models['p']

    log_q_0_x_0 = data['t']['log_q_x']
    x_0 = data['t']['x']
    y_0 = data['t']['y']
    theta = data['tm1']['theta']

    # print('x_0_online', x_0)

    p_emission_term = p.emission_kernel.logpdf(y_0, x_0, theta.emission)

    p_prior_term =  p.prior_dist.logpdf(x_0, theta.prior)
    q_initial_term = log_q_0_x_0
    result = p_emission_term + p_prior_term - q_initial_term
    # print('h_0_online', result)
    # print('p_terms_0_online', p_prior_term + p_emission_term)

    return result

def elbo_h_t_online(data, models):

    p = models['p']
    q = models['q']

    x_t = data['t']['x']
    # print('x_1_online', x_t)

    y_t = data['t']['y']
    
    x_tm1 = data['tm1']['x']
    theta = data['tm1']['theta']

    log_q_t_x_t = data['t']['log_q_x']
    log_q_tm1_x_tm1 = data['tm1']['log_q_x']
    log_q_tm1_t_x_tm1_x_t = data['tm1']['log_q_backwd_x']

    p_transition_term = p.transition_kernel.logpdf(x_t, x_tm1, theta.transition)
    p_emission_term = p.emission_kernel.logpdf(y_t, x_t, theta.emission)
    # print('p_emission_term_1_online', p_emission_term)
    result = p_transition_term \
            + p_emission_term \
            + log_q_tm1_x_tm1 \
            - log_q_t_x_t \
            - log_q_tm1_t_x_tm1_x_t
    
    # print('h_t_online', result)

    return result



def elbo_h_0_offline(data, models): 

    p = models['p']
    q = models['q']

    x_tp1 = data['tp1']['x']
    y_tp1 = data['tp1']['y']
    theta = data['tp1']['theta']

    x_t = data['t']['x']
    y_t = data['t']['y']
    params_q_t_tp1 = data['t']['params_backwd']
    
    p_prior_term = p.prior_dist.logpdf(x_t, theta.prior)
    p_transition_term = p.transition_kernel.logpdf(x_tp1, x_t, theta.transition)
    p_emission_terms = p.emission_kernel.logpdf(y_tp1, x_tp1, theta.emission) + p.emission_kernel.logpdf(y_t, x_t, theta.emission)
    q_backwd_term = q.backwd_kernel.logpdf(x_t, x_tp1, params_q_t_tp1)
    # print('x_0_offline', x_t)

    # print('p_terms_0_offline', p_prior_term + p.emission_kernel.logpdf(y_t, x_t, theta.emission))
    # print('p_emission_term_1_offline',  p.emission_kernel.logpdf(y_tp1, x_tp1, theta.emission))

    result =  p_prior_term \
        +  p_transition_term \
        + p_emission_terms \
        - q_backwd_term

    # print('h_0_offline', result)
    return result 



def elbo_h_t_offline(data, models):

    p = models['p']
    q = models['q']

    x_tp1 = data['tp1']['x']
    y_tp1 = data['tp1']['y']
    theta = data['tp1']['theta']

    x_t = data['t']['x']
    params_q_t_tp1 = data['t']['params_backwd']
    
    p_transition_term = p.transition_kernel.logpdf(x_tp1, x_t, theta.transition)
    p_emission_term = p.emission_kernel.logpdf(y_tp1, x_tp1, theta.emission)
    # print('p_emission_term_1_online', p_emission_term)
    q_backwd_term = q.backwd_kernel.logpdf(x_t, x_tp1, params_q_t_tp1)

    result =  p_transition_term \
            + p_emission_term\
            - q_backwd_term
    

    # print('h_t_offline', result)
    return result 

def elbo_h_T_offline(data, models):

    q = models['q']

    x_t = data['t']['x']
    params_q_t = data['t']['params_q']
    

    # print('x_1_offline', x_t)
    q_terminal_term = q.filt_dist.logpdf(x_t, params_q_t)

    result = -q_terminal_term

    # print('h_T_offline', result)
    return result 

def state_smoothing_h_t(data, models):
    return data['t']['x']



def x1_x2_functional_online_0(data, models):
    return jnp.zeros((models['p'].state_dim,))

def x1_x2_functional_online_t(data, models):
    return data['tm1']['x'] * data['t']['x']

def x1_x2_functional_offline_T(data, models):
    return jnp.zeros((models['p'].state_dim,))

def x1_x2_functional_offline_t(data, models):
    return data['t']['x'] * data['tp1']['x']




def samples_and_log_probs(dist, key, params_dist, num_samples):
    samples = vmap(dist.sample, in_axes=(0, None))(random.split(key, num_samples), params_dist)
    log_probs = vmap(dist.logpdf, in_axes=(0, None))(samples, params_dist)
    return samples, log_probs

class AdditiveFunctional:

    def __init__(self, h_t, out_shape, h_0=None, h_T=None):
        
        self.out_shape = out_shape
        
        self.update = h_t 
        self.init = h_t if h_0 is None else h_0
        self.end = h_t if h_T is None else h_T



online_elbo_functional = lambda p, q: AdditiveFunctional(h_t=elbo_h_t_online, 
                                                        out_shape=(), 
                                                        h_0=elbo_h_0_online)


offline_elbo_functional = lambda p, q: AdditiveFunctional(h_0=elbo_h_0_offline, 
                                                        h_t=elbo_h_t_offline, 
                                                        h_T=elbo_h_T_offline, 
                                                        out_shape=())


state_smoothing_functional = lambda p, q: AdditiveFunctional(h_t=state_smoothing_h_t, 
                                                            out_shape=(p.state_dim,))

online_x1_x2_functional = lambda p, q: AdditiveFunctional(h_0=x1_x2_functional_online_0, 
                                                        out_shape=(p.state_dim,),
                                                        h_t=x1_x2_functional_online_t)

            
offline_x1_x2_functional = lambda p, q: AdditiveFunctional(h_T=x1_x2_functional_offline_T, 
                                                        out_shape=(p.state_dim,),
                                                        h_t=x1_x2_functional_offline_t)                                            


def nested_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def named_vmap(f, axes_names, input_dict):    

    in_axes = nested_dict_update(tree_util.tree_map(lambda x: None, input_dict), 
                                axes_names)

    return jax.vmap(f, in_axes=(in_axes,))(input_dict)



def get_defaults(args):
    import math
    args.float64 = True

    args.default_prior_mean = 0.0 # default value for the mean of Gaussian prior
    args.default_prior_base_scale = math.sqrt(1e-2) # default value for the diagonal components of the covariance matrix of the prior
    args.default_transition_base_scale = math.sqrt(1e-1) # default value for the diagonal components of the covariance matrix of the transition kernel
    args.default_transition_bias = 0.0
    args.default_emission_base_scale = math.sqrt(1e-1)


    if args.model == 'linear' and (not hasattr(args, 'emission_bias')):
        args.emission_bias = False
        
    if 'chaotic_rnn' in args.model:
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
        args.transition_bias = False 
        args.emission_bias = False

    if 'nonlinear_emission' in args.model:
        args.emission_map_layers = (8,)
        args.slope = 0 # amount of linearity in the emission function
        args.injective = True

    if 'neural_backward' or 'johnson' in args.model:
        ## variational family
        args.update_layers = (100,) # number of layers in the GRU which updates the variational filtering dist
        # args.backwd_map_layers = (32,) # number of layers in the MLP which predicts backward parameters (not used in the Johnson method)

    if 'johnson' in args.model:
        args.anisotropic = 'anisotropic' in args.model

    if 'neural_backward' in args.model:
        if not 'explicit_transition' in args.model_options:
            args.backwd_layers = (100,)
        else: 
            args.backwd_layers = 0

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

    slice_args = lambda a, start, stop: lax.cond(stop!=-1, 
                                                lambda a, start, stop: (start, stop-start), 
                                                lambda a, start, stop: (start, a.shape[0] - start),
                                                a, start, stop)

    return tree_map(lambda a: lax.dynamic_slice_in_dim(a, *slice_args(a, start, stop)), tree)


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


def normalize(x):
    w = jnp.exp(x)

    return w / w.sum()

def cosine_similarity(oracle, estimate):
    return (oracle @ estimate) / (jnp.linalg.norm(oracle, ord=2) \
                                  * jnp.linalg.norm(estimate, ord=2))




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
