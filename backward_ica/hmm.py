from abc import ABCMeta, abstractmethod
from re import A
from turtle import back
from jax import numpy as jnp, random, tree_flatten, tree_leaves
from backward_ica.kalman import kalman_init, kalman_predict, kalman_update, kalman_filter_seq
from backward_ica.smc import smc_compute_filt_seq, smc_filter_seq, smc_smooth_from_filt_seq
import haiku as hk
from jax import lax, vmap, config
config.update('jax_enable_x64', True)
from .utils import *
from jax.scipy.stats.multivariate_normal import logpdf as gaussian_logpdf
from jax.tree_util import Partial
from functools import partial
from jax import nn
import jax
import optax 



def symetric_def_pos(param):
    tril_mat = jnp.zeros()
    return param @ param.T

def mean_cov_from_vec(vec, d):

    mean = vec[:d]
    cov_chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(vec[d:])
    return mean, cov_chol @ cov_chol.T

def mean_cov_chol_from_vec(vec, d):
    mean = vec[:d]
    cov_chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(vec[d:])
    return mean, cov_chol 

def vec_from_mean_cov(mean, cov):
    return jnp.concatenate((mean, jnp.tril(jnp.linalg.cholesky(cov)).flatten()))

_conditionnings = {'diagonal':lambda param: jnp.diag(param),
                'symetric_def_pos': lambda param: param @ param.T,
                None:lambda x:x}



def xtanh(slope):
    return lambda x: jnp.tanh(x) + slope*x

def neural_map(input, hidden_layer_sizes, slope, out_dim):

    net = hk.nets.MLP((*hidden_layer_sizes, out_dim), 
                    with_bias=False, 
                    activate_final=True, 
                    w_init=hk.initializers.Orthogonal(),
                    activation=xtanh(slope))

    return net(input)

# def filt_predict_forward(shared_params, filt_state, out_dim):
#     net = hk.nets.MLP([8, out_dim], w_init=hk.initializers.Orthogonal())
#     return net(jnp.concatenate((shared_params, filt_state)))


def forget_gate_forward(obs, filt_state, out_dim):

    net = hk.nets.MLP((out_dim,), 
                    activation=nn.sigmoid, 
                    w_init=hk.initializers.Orthogonal(),
                    b_init=hk.initializers.Constant(1),
                    activate_final=True)

    return net(jnp.concatenate((obs, filt_state)))


def filt_update_forward(obs, pred_state, out_dim):
    net = hk.nets.MLP((100,8,out_dim), 
                    w_init=hk.initializers.Orthogonal(),
                    activate_final=True)
    return net(jnp.concatenate((obs, pred_state)))

# def backwd_update_forward(hidden_filt_state, out_dim):
#     net = hk.nets.MLP([2, out_dim], w_init=hk.initializers.Orthogonal())
#     return net(hidden_filt_state)

def linear_map_apply(params, input):
    out = jnp.dot(params.w, input)
    return out + jnp.broadcast_to(params.b, out.shape)

def linear_map_init_params(key, dummy_in, out_dim, conditionning):
    key, subkey = random.split(key, 2)

    if conditionning == 'diagonal':
        w = random.uniform(key, (out_dim,))
    else: 
        w = random.uniform(key, (out_dim, len(dummy_in)))
    
    b = random.uniform(subkey, (out_dim,))

    return LinearMapParams(w, b)

def linear_map_format_params(params, conditionning_func):

    return LinearMapParams(conditionning_func(params.w), params.b)


class Gaussian: 

    @staticmethod
    def sample(key, params):
        return params.mean + params.scale.chol @ random.normal(key, (params.mean.shape[0],))
    
    @staticmethod
    def logpdf(x, params):
        return gaussian_logpdf(x, params.mean, params.scale.cov)

    @staticmethod
    def get_random_params(key, dim, default_base_scale):
        mean = random.uniform(key, shape=(dim,)) 
        scale = default_base_scale * jnp.ones((dim,))
        return GaussianParams(mean=mean, scale=scale)

    @staticmethod
    def format_params(params):
        return GaussianParams(params.mean, Scale(chol=jnp.diag(params.scale)))

@register_pytree_node_class
class Kernel:

    def __init__(self,
                in_dim, 
                out_dim,
                map_def, 
                noise_dist=Gaussian):

        self.in_dim = in_dim
        self.out_dim = out_dim 
        map_name, map_args = map_def

        if map_name == 'linear':
            conditionning = map_args
            apply_map = linear_map_apply
            init_map_params = partial(linear_map_init_params, out_dim=out_dim, conditionning=conditionning)
            format_map_params = partial(linear_map_format_params, conditionning_func=_conditionnings[conditionning])
        else: 
            hidden_layer_sizes, slope = map_args 
            init_map_params, apply_map = hk.without_apply_rng(hk.transform(partial(neural_map, out_dim=out_dim, hidden_layer_sizes=hidden_layer_sizes, slope=slope)))
            format_map_params = lambda x:x

        self.apply_map, self.init_map_params, self.format_map_params =  Partial(apply_map), Partial(init_map_params), Partial(format_map_params)
        self.noise_sample, self.noise_logpdf = Partial(noise_dist.sample), Partial(noise_dist.logpdf)
        
    def map(self, state, params):
        return self.apply_map(params.map, state)
    
    def sample(self, key, state, params):
        return self.noise_sample(key, GaussianParams(self.map(state, params), params.scale))

    def logpdf(self, x, state, params):
        return self.noise_logpdf(x, GaussianParams(self.map(state, params), params.scale))

    def get_random_params(self, key, default_base_scale=None):
        subkeys = random.split(key, 2)
        return KernelParams(map=self.init_map_params(subkeys[0], jnp.empty((self.in_dim,))),
                            scale=default_base_scale * jnp.ones((self.out_dim,)))
    
    def format_params(self, params):
        return KernelParams(map=self.format_map_params(params.map),
                            scale=Scale(chol=jnp.diag(params.scale)))
    
    def tree_flatten(self):
        return ((self.in_dim, self.out_dim, self.apply_map, self.init_map_params, self.format_map_params, self.noise_sample, self.noise_logpdf), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.in_dim, obj.out_dim, obj.apply_map, obj.init_map_params, obj.format_map_params, obj.noise_sample, obj.noise_logpdf = children
        return obj

class HMM: 

    default_prior_base_scale = 5e-2
    default_transition_base_scale = 5e-2
    default_emission_base_scale = 2e-2

    def __init__(self, 
                state_dim, 
                obs_dim, 
                prior_dist,
                transition_kernel_type, 
                emission_kernel_type):

        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.prior_dist:Gaussian = prior_dist
        self.transition_kernel:Kernel = transition_kernel_type(state_dim)
        self.emission_kernel:Kernel = emission_kernel_type(state_dim, obs_dim)
        
    def sample_multiple_sequences(self, key, params, num_seqs, seq_length):
        key, *subkeys = random.split(key, num_seqs+1)
        sampler = vmap(self.sample_seq, in_axes=(0, None, None))
        return sampler(jnp.array(subkeys), params, seq_length)

    def get_random_params(self, key):

        key_prior, key_transition, key_emission = random.split(key, 3)
        prior_params = self.prior_dist.get_random_params(key_prior, self.state_dim, self.default_prior_base_scale)
        transition_params = self.transition_kernel.get_random_params(key_transition, self.default_transition_base_scale)
        emission_params = self.emission_kernel.get_random_params(key_emission, self.default_emission_base_scale)
        return HMMParams(prior_params, transition_params, emission_params)
        
    def format_params(self, params):

        return HMMParams(self.prior_dist.format_params(params.prior),
                        self.transition_kernel.format_params(params.transition),
                        self.emission_kernel.format_params(params.emission))
                        
    
    def sample_seq(self, key, params, seq_length):

        params = self.format_params(params)

        keys = random.split(key, 2*seq_length)
        state_keys = keys[:seq_length]
        obs_keys = keys[seq_length:]

        prior_sample = self.prior_dist.sample(state_keys[0], params.prior)

        def _state_sample(carry, x):
            prev_sample = carry
            key = x
            sample = self.transition_kernel.sample(key, prev_sample, params.transition)
            return sample, sample
        _, state_seq = lax.scan(_state_sample, init=prior_sample, xs=state_keys[1:])

        state_seq = tree_prepend(prior_sample, state_seq)
        obs_seq = vmap(self.emission_kernel.sample, in_axes=(0,0,None))(obs_keys, state_seq, params.emission)

        return state_seq, obs_seq
        
    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', sum(len(leaf) for leaf in tree_leaves(params)))
        print('-- in prior + predict:', sum(len(leaf) for leaf in tree_leaves((params.prior, params.transition))))
        print('-- in update:', sum(len(leaf) for leaf in tree_leaves(params.emission)))

class BackwardSmoother(metaclass=ABCMeta):

    def __init__(self, backwd_kernel, filt_dist):
        self.filt_dist = filt_dist
        self.backwd_kernel:Kernel = backwd_kernel

    @abstractmethod
    def get_random_params(self, key):
        raise NotImplementedError

    @abstractmethod
    def format_params(self, params):
        raise NotImplementedError

    @abstractmethod
    def init_filt_state(self, obs, params, *args):
        raise NotImplementedError

    @abstractmethod
    def new_filt_state(self, obs, filt_state, params, *args):
        raise NotImplementedError

    @abstractmethod
    def new_backwd_state(self, filt_state, params, *args):
        raise NotImplementedError

    @abstractmethod
    def compute_filt_seq(self, *args):
        raise NotImplementedError

    @abstractmethod
    def compute_backwd_seq(self, *args):
        raise NotImplementedError

    @abstractmethod
    def backwd_pass(self, last_filt_state, backwd_state_seq):
        raise NotImplementedError

    def smooth_seq(self, obs_seq, params, *args):

        filt_state_seq = self.compute_filt_seq(obs_seq, params)
        backwd_state_seq = self.compute_backwd_seq(filt_state_seq, params)

        return self.backwd_pass(tree_get_idx(-1, filt_state_seq), backwd_state_seq)

class LinearGaussianHMM(HMM, BackwardSmoother):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning):

        transition_kernel_type = lambda state_dim:Kernel(state_dim, state_dim, ('linear', transition_matrix_conditionning))
        emission_kernel_type = lambda state_dim, obs_dim:Kernel(state_dim, obs_dim, ('linear', None))

        HMM.__init__(self, 
                    state_dim, 
                    obs_dim, 
                    Gaussian, 
                    transition_kernel_type, 
                    emission_kernel_type)

        BackwardSmoother.__init__(self, backwd_kernel=Kernel(state_dim, state_dim, ('linear', None)), filt_dist=Gaussian)

    def init_filt_state(self, obs, params):

        mean, cov =  kalman_init(obs, params.prior.mean, params.prior.scale.cov, params.emission)

        return GaussianParams(mean, Scale(cov=cov))

    def new_filt_state(self, obs, filt_state, params):
        pred_mean, pred_cov = kalman_predict(filt_state.mean, filt_state.scale.cov, params.transition)
        mean, cov = kalman_update(pred_mean, pred_cov, obs, params.emission)

        return GaussianParams(mean, Scale(cov=cov))

    def new_backwd_state(self, filt_state, params):

        A, a, Q = *params.transition.map, params.transition.scale.cov
        mu, Sigma = filt_state.mean, filt_state.scale.cov
        I = jnp.eye(self.state_dim)

        k_chol_inv = inv_of_chol(A @ Sigma @ A.T)
        K = Sigma @ A.T @ k_chol_inv @ jnp.linalg.inv(k_chol_inv @ Q @ k_chol_inv + I) @ k_chol_inv

        C = I - K @ A

        A_back = K 
        a_back = mu @ C - K @ a
        cov_back = C @ Sigma

        return KernelParams(LinearMapParams(A_back, a_back), Scale(cov=cov_back))

    def likelihood_seq(self, obs_seq, params, *args):
        return kalman_filter_seq(obs_seq, self.format_params(params))[-1]
    
    def compute_filt_seq(self, obs_seq, params, *args):

        return kalman_filter_seq(obs_seq, self.format_params(params))[2:4]

    def compute_backwd_seq(self, filt_seq, params, *args):
        
        params = self.format_params(params)

        def backwd_from_filt(filt_state):

            A, a, Q = *params.transition.map, params.transition.scale.cov
            I = jnp.eye(self.state_dim)
            mu, Sigma = filt_state

            k_chol_inv = inv_of_chol(A @ Sigma @ A.T)
            K = Sigma @ A.T @ k_chol_inv @ jnp.linalg.inv(k_chol_inv @ Q @ k_chol_inv + I) @ k_chol_inv

            C = I - K @ A

            A_back = K 
            a_back = mu @ C - K @ a
            cov_back = C @ Sigma
            return A_back, a_back, cov_back
            
        return vmap(backwd_from_filt)(tree_droplast(filt_seq))

    def backwd_pass(self, last_filt_state, backwd_state_seq):
        last_filt_state_mean, last_filt_state_cov = last_filt_state

        @jit
        def _step(filt_state, backwd_state):
            A_back, a_back, cov_back = backwd_state
            filt_state_mean, filt_state_cov = filt_state
            mean = A_back @ filt_state_mean + a_back
            cov = A_back @ filt_state_cov @ A_back.T + cov_back
            return (mean, cov), (mean, cov)

        means, covs = lax.scan(_step, 
                                init=(last_filt_state_mean, last_filt_state_cov), 
                                xs=backwd_state_seq, 
                                reverse=True)[1]
        
        return tree_append(means, last_filt_state_mean), tree_append(covs, last_filt_state_cov) 

    def gaussianize_filt_state(self, filt_state, params):
        return filt_state

class NonLinearGaussianHMM(HMM):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning,
                hidden_layer_sizes,
                slope):

        transition_kernel_type = lambda state_dim: Kernel(state_dim, state_dim, ('linear', transition_matrix_conditionning))
        emission_kernel_type = lambda state_dim, obs_dim:Kernel(state_dim, obs_dim, ('nonlinear', (hidden_layer_sizes, slope)))

        HMM.__init__(self, state_dim, obs_dim, Gaussian, transition_kernel_type, emission_kernel_type)

    def likelihood_seq(self, obs_seq, params, key, num_particles):

        return smc_filter_seq(key, 
                            obs_seq, 
                            self.format_params(params),
                            Partial(self.prior_dist.sample), 
                            self.transition_kernel, 
                            self.emission_kernel,
                            num_particles)[-1]
    
    def compute_filt_seq(self, obs_seq, params, key, num_particles):

        return smc_compute_filt_seq(key, 
                                obs_seq, 
                                self.format_params(params), 
                                Partial(self.prior_dist.sample), 
                                self.transition_kernel, 
                                self.emission_kernel, 
                                num_particles)
        
    def backwd_pass(self, filt_seq, params, key):
        formatted_params = self.format_params(params)
        return smc_smooth_from_filt_seq(key, filt_seq, formatted_params, self.transition_kernel)
    
    def smooth_seq(self, obs_seq, params, key, num_particles):
        filt_seq = self.compute_filt_seq(obs_seq, params, key, num_particles)
        return self.backwd_pass(filt_seq, params, key)

class NeuralBackwardSmoother(BackwardSmoother):

    def __init__(self, state_dim, obs_dim, backwd_kernel_type='linear', filt_dist=Gaussian):
        
        if backwd_kernel_type == 'linear':
            backwd_kernel = Kernel(state_dim, state_dim, ('linear', None), Gaussian)
            self.transition_kernel = Kernel(state_dim, state_dim, ('linear', 'diagonal'), Gaussian)

        super().__init__(backwd_kernel, filt_dist)

        self.state_dim, self.obs_dim = state_dim, obs_dim 
        d = self.state_dim
        self.d = d 
        self.filt_state_shape = d + d*(d+1) // 2
        

        self.forget_gate_init_params, self.forget_apply = hk.without_apply_rng(hk.transform(partial(forget_gate_forward, out_dim=self.filt_state_shape)))
        self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(filt_update_forward, out_dim=self.filt_state_shape)))
        
        # self.backwd_update_init_params, self.backwd_uspdate_apply = hk.without_apply_rng(hk.transform(partial(backwd_update_forward, out_dim=d**2 + self.filt_state_shape)))

    def get_random_params(self, key):

        subkeys = random.split(key, 4)

        dummy_obs = jnp.empty((self.obs_dim,))
        prior_params = GaussianBaseParams(mean=random.uniform(subkeys[0], (self.state_dim,)), 
                                        base_scale=GaussianHMM.default_prior_base_scale * jnp.ones((self.state_dim,)))

        transition_params = self.transition_kernel.get_random_params(subkeys[1], GaussianHMM.default_transition_base_scale)

        filt_update_params = self.filt_update_init_params(subkeys[2], dummy_obs, jnp.empty((self.filt_state_shape,)))
        forget_gate_params = self.forget_gate_init_params(subkeys[3], dummy_obs, jnp.empty((self.filt_state_shape,)))

        return NeuralSmootherParams(prior_params, transition_params, filt_update_params, forget_gate_params)

    def init_filt_state(self, obs, params):

        filt_state = jnp.concatenate((params.prior.mean, jnp.tril(jnp.diag(params.prior.base_scale)).flatten()))

        forget_state = self.forget_apply(params.forget_gate, obs, filt_state)

        candidate_filt_state = self.filt_update_apply(params.filt_update, obs, filt_state)

        return (1 - forget_state) * filt_state + forget_state * candidate_filt_state


    def new_filt_state(self, obs, filt_state, params):

        pred_state =  vec_from_mean_cov(*kalman_predict(*mean_cov_from_vec(filt_state, self.d), 
                                                        params.transition))

        forget_state = self.forget_apply(params.forget_gate, obs, filt_state)
        
        candidate_filt_state  = self.filt_update_apply(params.filt_update, obs, pred_state)

        return (1 - forget_state) * filt_state + forget_state * candidate_filt_state
    
    def new_backwd_state(self, filt_state, params):

        A, a, Q = params.transition.matrix, params.transition.bias, params.transition.cov

        mu, Sigma = mean_cov_from_vec(filt_state, self.d)
        I = jnp.eye(self.state_dim)

        k_chol_inv = inv_of_chol(A @ Sigma @ A.T)
        K = Sigma @ A.T @ k_chol_inv @ jnp.linalg.inv(k_chol_inv @ Q @ k_chol_inv + I) @ k_chol_inv

        C = I - K @ A

        A_back = K 
        a_back = mu @ C - K @ a
        cov_back = C @ Sigma

        return BackwardParams(A_back, a_back, cov_back, log_det_from_cov(cov_back))

    # def new_backwd_state(self, filt_state, params):

    #     state = self.backwd_update_apply(params.backwd_update, filt_state.hidden)

    #     A_back = state[:self.d**2].reshape((self.d,self.d))
    #     a_back = state[self.d**2:self.d**2+self.d]
    #     cov_chol_back = jnp.zeros((self.d,self.d)).at[jnp.tril_indices(self.d)].set(state[self.d**2+self.d:])

    #     return BackwardParams(A_back, a_back, cov_chol_back @ cov_chol_back.T, log_det_from_chol(cov_chol_back))

    def compute_filt_seq(self, obs_seq, params):

        params = self.format_params(params)
        
        init_filt_state = self.init_filt_state(obs_seq[0], params)

        @jit
        def _step(carry, x):
            filt_state, params = carry
            obs = x
            filt_state = self.new_filt_state(obs, filt_state, params)
            return (filt_state, params), filt_state

        filt_state_seq = lax.scan(_step, init=(init_filt_state, params), xs=obs_seq[1:])[1]

        filt_state_seq =  tree_prepend(init_filt_state, filt_state_seq)

        return vmap(mean_cov_from_vec, in_axes=(0,None))(filt_state_seq, self.d)

    def compute_backwd_seq(self, filt_seq, params):
        
        params = self.format_params(params)

        def backwd_from_filt(filt_state):
            mu, Sigma = filt_state
            A, a, Q = params.transition.matrix, params.transition.bias, params.transition.cov
            I = jnp.eye(self.state_dim)

            k_chol_inv = inv_of_chol(A @ Sigma @ A.T)
            K = Sigma @ A.T @ k_chol_inv @ jnp.linalg.inv(k_chol_inv @ Q @ k_chol_inv + I) @ k_chol_inv

            C = I - K @ A

            A_back = K 
            a_back = mu @ C - K @ a
            cov_back = C @ Sigma
            return A_back, a_back, cov_back
            
        return vmap(backwd_from_filt)(tree_droplast(filt_seq))

    def backwd_pass(self, last_filt_state, backwd_state_seq):
        last_filt_state_mean, last_filt_state_cov = last_filt_state

        @jit
        def _step(filt_state, backwd_state):
            A_back, a_back, cov_back = backwd_state
            filt_state_mean, filt_state_cov = filt_state
            mean = A_back @ filt_state_mean + a_back
            cov = A_back @ filt_state_cov @ A_back.T + cov_back
            return (mean, cov), (mean, cov)

        means, covs = lax.scan(_step, 
                                init=(last_filt_state_mean, last_filt_state_cov), 
                                xs=backwd_state_seq, 
                                reverse=True)[1]
        
        return tree_append(means, last_filt_state_mean), tree_append(covs, last_filt_state_cov) 

    def gaussianize_filt_state(self, filt_state, params):
        mean = filt_state[:self.d]
        cov_chol = jnp.zeros((self.d,self.d)).at[jnp.tril_indices(self.d)].set(filt_state[self.d:])
        return FiltParams(filt_state, mean, cov_chol @ cov_chol.T, log_det_from_chol(cov_chol))

    def format_params(self, params):
        formatted_transition_params = self.transition_kernel.format_params(params.transition)
        return NeuralSmootherParams(params.prior, formatted_transition_params, params.filt_update, params.forget_gate)

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', sum(len(leaf) for leaf in tree_leaves(params)))
        print('-- in prior + predict + backward:', sum(len(leaf) for leaf in tree_leaves((params.prior, params.transition))))
        print('-- in update:', sum(len(leaf) for leaf in tree_leaves(params.filt_update)))
        print('-- in forget gate:', sum(len(leaf) for leaf in tree_leaves(params.forget_gate)))

