from abc import ABCMeta, abstractmethod
from atexit import register
from .utils import *
from jax import numpy as jnp, random
from backward_ica.kalman import kalman_init, kalman_predict, kalman_smooth_seq, kalman_update, kalman_filter_seq
from backward_ica.smc import smc_filter_seq, smc_smooth_seq
import haiku as hk
from jax import lax, vmap, config
config.update('jax_enable_x64', True)
from .utils import *
from jax.scipy.stats.multivariate_normal import logpdf as gaussian_logpdf
from jax.tree_util import Partial
from functools import partial
from jax import nn

def sample_multiple_sequences(key, hmm_sampler, hmm_params, num_seqs, seq_length):
    key, *subkeys = random.split(key, num_seqs+1)
    sampler = vmap(hmm_sampler, in_axes=(0, None, None))
    return sampler(jnp.array(subkeys), hmm_params, seq_length) 

def symetric_def_pos(param):
    tril_mat = jnp.zeros()
    return param @ param.T


_conditionnings = {'diagonal':lambda param: jnp.diag(param),
                'symetric_def_pos': lambda param: param @ param.T,
                None:lambda x:x}


def linear_map(input, params):
    return params.matrix @ input + params.bias 

def neural_map(input, out_dim):
    net = hk.nets.MLP((8, out_dim), activate_final=True, activation=nn.sigmoid)
    return net(input)

def filt_predict_forward(shared_params, filt_state, out_dim):
    filt_mean, filt_cov = filt_state
    net = hk.nets.MLP([8, out_dim])
    out = net(jnp.concatenate((shared_params, filt_mean, jnp.tril(filt_cov).flatten())))
    return out

def filt_update_forward(pred_state, obs, d):
    net = hk.nets.MLP([8, d + d*(d+1) // 2])
    out = net(jnp.concatenate((obs, pred_state)))
    mean = out[:d]
    cov_chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(out[d:])
    return mean, cov_chol @ cov_chol.T

def backwd_update_forward(shared_params, filt_state, d):
    filt_mean, filt_cov = filt_state
    net = hk.nets.MLP([8, d**2 + d + d*(d+1) // 2])
    out = net(jnp.concatenate((shared_params, filt_mean, jnp.tril(filt_cov).flatten())))
    A = out[:d**2].reshape((d,d))
    a = out[d**2:d**2+d]
    cov_chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(out[d**2+d:])
    return A, a, cov_chol @ cov_chol.T



@register_pytree_node_class
class GaussianKernel:

    def __init__(self, in_dim, out_dim):

        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.map_init_params, self.map_apply = hk.without_apply_rng(hk.transform(partial(neural_map, out_dim=self.out_dim)))
        self.map_init_params = Partial(self.map_init_params)
        self.map_apply = Partial(self.map_apply)

    def map(self, state, params):
        return self.map_apply(params.map_params, state)
    
    def sample(self, key, mapped_state, params):
        return random.multivariate_normal(key, mapped_state, params.cov)

    def logpdf(self, x, mapped_state, params):
        return gaussian_logpdf(x, mapped_state, params.cov)

    def get_random_params(self, key, default_cov_base=None):
        subkeys = random.split(key, 2)
        return GaussianKernelBaseParams(map_params=self.map_init_params(subkeys[0], jnp.empty((self.in_dim,))),
                                                cov_base=default_cov_base * jnp.ones((self.out_dim,)))
    
    def format_params(self, params):
        return GaussianKernelParams(params.map_params,
                                    *cov_params_from_cov_chol(jnp.diag(params.cov_base)))
    
    def tree_flatten(self):
        return ((self.in_dim, self.out_dim, self.map_init_params, self.map_apply), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.in_dim, obj.out_dim, obj.map_init_params, obj.map_apply = children
        return obj

@register_pytree_node_class
class LinearGaussianKernel(GaussianKernel):

    def __init__(self, in_dim, out_dim, matrix_conditionning=None):

        self.in_dim = in_dim 
        self.out_dim = out_dim
        self.matrix_conditionning = matrix_conditionning
        self.map_apply = Partial(vmap(linear_map, in_axes=(0,None)))

    def map(self, state, params):
        return self.map_apply(state, params)

    def get_random_params(self, key, default_cov_base=None):
        subkeys = random.split(key, 3)

        if self.matrix_conditionning == 'diagonal':
            matrix = random.uniform(subkeys[0], shape=(self.in_dim,))
        else: 
            matrix = random.uniform(subkeys[0], shape=(self.out_dim, self.in_dim))


        return LinearGaussianKernelBaseParams(matrix=matrix,
                    bias=random.uniform(subkeys[1], shape=(self.out_dim,)),
                    cov_base=default_cov_base * jnp.ones((self.out_dim,)))

    def format_params(self, params):
        return LinearGaussianKernelParams(_conditionnings[self.matrix_conditionning](params.matrix),
                                        params.bias, 
                                        *cov_params_from_cov_chol(jnp.diag(params.cov_base)))

    def tree_flatten(self):
        return ((self.in_dim, self.out_dim, self.matrix_conditionning, self.map_apply), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.in_dim, obj.out_dim, obj.matrix_conditionning, obj.map_apply = children


class GaussianHMM: 

    default_prior_cov_base = 5e-2
    default_transition_cov_base = 5e-2
    default_emission_cov_base = 2e-2

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_kernel_type, 
                emission_kernel_type):

        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.prior_sampler = Partial(lambda key, params: random.multivariate_normal(key, params.mean, params.cov))
        self.transition_kernel:GaussianKernel = transition_kernel_type(state_dim)
        self.emission_kernel:GaussianKernel = emission_kernel_type(state_dim, obs_dim)
        
    def get_random_params(self, key):

        key, *subkeys = random.split(key, 3)
        prior_params = GaussianBaseParams(mean=random.uniform(subkeys[0], shape=(self.state_dim,)), 
                                    cov_base=GaussianHMM.default_prior_cov_base * jnp.ones((self.state_dim,)))
        subkeys = random.split(key, 2)

        transition_params = self.transition_kernel.get_random_params(subkeys[0], GaussianHMM.default_transition_cov_base)
        emission_params = self.emission_kernel.get_random_params(subkeys[1], GaussianHMM.default_emission_cov_base)
        return HMMParams(prior_params, transition_params, emission_params)
        
    def format_params(self, params):
        formatted_prior_params = GaussianParams(params.prior.mean, 
                                                *cov_params_from_cov_chol(jnp.diag(params.prior.cov_base)))

        formatted_transition_params = self.transition_kernel.format_params(params.transition)
        formatted_emission_params = self.emission_kernel.format_params(params.emission)

        return HMMParams(formatted_prior_params, formatted_transition_params, formatted_emission_params)
    
    def sample_seq(self, key, params, seq_length):

        params = self.format_params(params)

        keys = random.split(key, 2*seq_length)
        state_keys = keys[:seq_length]
        obs_keys = keys[seq_length:]

        prior_sample = jnp.atleast_2d(self.prior_sampler(state_keys[0], params.prior))

        def _state_sample(carry, x):
            prev_sample = carry
            key = x
            mapped_sample = self.transition_kernel.map(prev_sample, params.transition)
            sample = self.transition_kernel.sample(key, mapped_sample, params.transition)
            return sample, sample
        _, state_seq = lax.scan(_state_sample, init=prior_sample, xs=state_keys[1:])

        state_seq = jnp.concatenate((prior_sample[None,:], state_seq)).reshape(seq_length, self.state_dim)

        mapped_state_seq = self.emission_kernel.map(state_seq, params.emission)
        obs_seq = vmap(lambda key, mapped_state: self.emission_kernel.sample(key, mapped_state, params.emission))(obs_keys, mapped_state_seq)

        return state_seq, obs_seq

class Smoother(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def get_random_params(self, key):
        raise NotImplementedError

    @abstractmethod
    def format_params(self, params):
        raise NotImplementedError

    @abstractmethod
    def init_filt_state(self, obs, params):
        raise NotImplementedError

    @abstractmethod
    def new_filt_state(self, obs, filt_state, params):
        raise NotImplementedError

    @abstractmethod
    def new_backwd_state(self, filt_state, params):
        raise NotImplementedError

    def backwd_pass(self, last_filt_state, backwd_state_seq):
        last_filt_state_mean, last_filt_state_cov = last_filt_state.mean, last_filt_state.cov
        def _backwd_step(filt_state, backwd_state):
            filt_state_mean, filt_state_cov = filt_state
            mean = backwd_state.matrix @ filt_state_mean + backwd_state.bias
            cov = backwd_state.matrix @ filt_state_cov @ backwd_state.matrix.T + backwd_state.cov
            return (mean, cov), (mean, cov)

        _, (means, covs) = lax.scan(_backwd_step, 
                                init=(last_filt_state_mean, last_filt_state_cov), 
                                xs=backwd_state_seq, 
                                reverse=True)

        means = jnp.concatenate([means, last_filt_state_mean[None,:]])
        covs = jnp.concatenate([covs, last_filt_state_cov[None,:]])
        
        return means, covs 

    def smooth_seq(self, obs_seq, params):
        # return kalman_smooth_seq(obs_seq, self.format_params(params))

        formatted_params = self.format_params(params)
        filt_state = self.init_filt_state(obs_seq[0], formatted_params)

        def _forward_pass(carry, x):
            filt_state, params = carry 
            obs = x 
            backwd_state = self.new_backwd_state(filt_state, params)
            filt_state = self.new_filt_state(obs, filt_state, params)
            return (filt_state, params), backwd_state

        (last_filt_state, _), backwd_state_seq = lax.scan(_forward_pass, init=(filt_state, formatted_params), xs=obs_seq[1:])

        return self.backwd_pass(last_filt_state, backwd_state_seq)


class LinearGaussianHMM(GaussianHMM, Smoother):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning):

        transition_kernel_type = lambda state_dim:LinearGaussianKernel(state_dim, state_dim, transition_matrix_conditionning)
        emission_kernel_type = LinearGaussianKernel

        GaussianHMM.__init__(self, state_dim, obs_dim, transition_kernel_type, emission_kernel_type)
        Smoother.__init__(self)

    def init_filt_state(self, obs, params):

        mean, cov =  kalman_init(obs, params.prior.mean, params.prior.cov, params.emission)

        return GaussianParams(mean, *cov_params_from_cov(cov))

    def new_filt_state(self, obs, filt_state, params):
        pred_mean, pred_cov = kalman_predict(filt_state.mean, filt_state.cov, params.transition)
        mean, cov = kalman_update(pred_mean, pred_cov, obs, params.emission)

        return GaussianParams(mean, *cov_params_from_cov(cov))

    def new_backwd_state(self, filt_state:GaussianParams, params):

        A, a, Q = params.transition.matrix, params.transition.bias, params.transition.cov
        mu, Sigma = filt_state.mean, filt_state.cov
        I = jnp.eye(self.state_dim)

        k_chol_inv = inv_of_chol(A @ Sigma @ A.T)
        K = Sigma @ A.T @ k_chol_inv @ jnp.linalg.inv(k_chol_inv @ Q @ k_chol_inv + I) @ k_chol_inv

        C = I - K @ A

        A_back = K 
        a_back = mu @ C - K @ a
        cov_back = C @ Sigma

        return LinearGaussianKernelParams(A_back, a_back, *cov_params_from_cov(cov_back))

    def likelihood_seq(self, obs_seq, params):
        return kalman_filter_seq(obs_seq, self.format_params(params))[-1]

class NonLinearGaussianHMM(GaussianHMM):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning):

        transition_kernel_type = lambda state_dim: LinearGaussianKernel(state_dim, state_dim, transition_matrix_conditionning)
        emission_kernel_type = GaussianKernel

        GaussianHMM.__init__(self, state_dim, obs_dim, transition_kernel_type, emission_kernel_type)

    def likelihood_seq(self, obs_seq, prior_keys, resampling_keys, proposal_keys, params, num_particles):

        return smc_filter_seq(prior_keys, 
                            resampling_keys, 
                            proposal_keys, 
                            obs_seq, 
                            self.prior_sampler, 
                            self.transition_kernel, 
                            self.emission_kernel, 
                            self.format_params(params),
                            num_particles)[-1]

    def smooth_seq(self, prior_keys, resampling_keys, proposal_keys, obs_seq, params, num_particles):
        
        return smc_smooth_seq(prior_keys, 
                        resampling_keys, 
                        proposal_keys, 
                        obs_seq, 
                        self.prior_sampler, 
                        self.transition_kernel, 
                        self.emission_kernel, 
                        self.format_params(params), 
                        num_particles)

    # def smooth_seq()


class NeuralSmoother(Smoother):

    def __init__(self, state_dim, obs_dim):

        self.state_dim, self.obs_dim = state_dim, obs_dim 
        d = self.state_dim
        self.shared_param_shape = 8

        self.filt_predict_init_params, self.filt_predict_apply = hk.without_apply_rng(hk.transform(partial(filt_predict_forward, out_dim=self.shared_param_shape)))
        self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(filt_update_forward, d=d)))
        self.backwd_update_init_params, self.backwd_update_apply = hk.without_apply_rng(hk.transform(partial(backwd_update_forward, d=d)))

        
    def get_random_params(self, key):

        subkeys = random.split(key, 5)
        prior_params = random.uniform(subkeys[0], shape=(self.shared_param_shape,))
        shared_params = random.uniform(subkeys[1], shape=(self.shared_param_shape,))
        dummy_mean = jnp.empty((self.state_dim,))
        dummy_cov = jnp.empty((self.state_dim, self.state_dim))
        dummy_obs = jnp.empty((self.obs_dim,))

        filt_predict_params = self.filt_predict_init_params(subkeys[2], shared_params, (dummy_mean, dummy_cov))
        filt_update_params = self.filt_update_init_params(subkeys[3], prior_params, dummy_obs)
        backwd_update_params = self.backwd_update_init_params(subkeys[4], shared_params, (dummy_mean, dummy_cov))

        return NeuralSmootherParams(prior_params, shared_params, filt_predict_params, filt_update_params, backwd_update_params)


    def init_filt_state(self, obs, params):

        mean, cov = self.filt_update_apply(params.filt_update, params.prior, obs)
        return GaussianParams(mean, *cov_params_from_cov(cov))


    def new_filt_state(self, obs, filt_state, params):
        pred_state = self.filt_predict_apply(params.filt_predict, params.shared, (filt_state.mean, filt_state.cov))
        mean, cov = self.filt_update_apply(params.filt_update, pred_state, obs)
        return GaussianParams(mean, *cov_params_from_cov(cov))
    
    def new_backwd_state(self, filt_state, params):

        A_back, a_back, cov_back = self.backwd_update_apply(params.backwd_update, params.shared, (filt_state.mean, filt_state.cov))

        return LinearGaussianKernelParams(A_back, a_back, *cov_params_from_cov(cov_back))

    def format_params(self, params):
        return params