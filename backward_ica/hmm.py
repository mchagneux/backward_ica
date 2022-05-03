from abc import ABCMeta, abstractmethod
from jax import numpy as jnp, random
from jax.tree_util import tree_leaves
from backward_ica.kalman import Kalman
from backward_ica.smc import SMC
import haiku as hk
from jax import lax, vmap
from .utils import *
from jax.scipy.stats.multivariate_normal import logpdf as gaussian_logpdf
from functools import partial
from jax import nn
from typing import Callable

def gaussian_params_from_vec(vec, d, chol_add=lambda d:jnp.zeros((d,d))):

    mean = vec[:d]
    chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(vec[d:])
    return GaussianParams(mean, scale=Scale(chol=chol + chol_add(d)))

def vec_from_gaussian_params(params, d):
    return jnp.concatenate((params.mean, params.scale.chol[jnp.tril_indices(d)]))

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

def linear_map_apply(map_params, input):
    out = jnp.dot(map_params.w, input)
    return out + jnp.broadcast_to(map_params.b, out.shape)

def linear_map_init_params(key, dummy_in, out_dim, conditionning):
    key, subkey = random.split(key, 2)

    if conditionning == 'diagonal':
        w = random.uniform(key, (out_dim,), minval=-1, maxval=1)
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
        mean = random.uniform(key, shape=(dim,), minval=-1, maxval=1) 
        scale = default_base_scale * jnp.ones((dim,))
        return GaussianParams(mean=mean, scale=scale)

    @staticmethod
    def format_params(params):
        return GaussianParams(params.mean, Scale(chol=jnp.diag(params.scale)))

class Kernel:


    def __init__(self,
                in_dim, 
                out_dim,
                kernel_def, 
                noise_dist=Gaussian):

        self.in_dim = in_dim
        self.out_dim = out_dim 
        kernel_type, kernel_args = kernel_def

        if kernel_type['map'] == 'linear':
            conditionning = kernel_args

            apply_map = lambda params, input: (linear_map_apply(params.map, input), params.scale)
            init_map_params = partial(linear_map_init_params, out_dim=out_dim, conditionning=conditionning)
            format_map_params = partial(linear_map_format_params, conditionning_func=_conditionnings[conditionning])

            def get_random_params(key, default_base_scale):
                key, subkey = random.split(key, 2)
                return KernelParams(map=init_map_params(key, jnp.empty((self.in_dim,))),
                                    scale=default_base_scale * jnp.ones((self.out_dim,)))

            def format_params(params):
                return KernelParams(map=format_map_params(params.map),
                            scale=Scale(chol=jnp.diag(params.scale)))



        else:
            if kernel_type['homogeneous']:
            
                map_forward = kernel_args 
                init_map_params, nonlinear_apply_map = hk.without_apply_rng(hk.transform(partial(map_forward, 
                                                                                    out_dim=out_dim)))
                apply_map = lambda params, input: (nonlinear_apply_map(params.map, input), params.scale)

                format_map_params = lambda x:x

                def get_random_params(key, default_base_scale):
                    key, subkey = random.split(key, 2)
                    return KernelParams(map=init_map_params(key, jnp.empty((self.in_dim,))),
                                        scale=default_base_scale * jnp.ones((self.out_dim,)))

                def format_params(params):
                    return KernelParams(map=format_map_params(params.map),
                                scale=Scale(chol=jnp.diag(params.scale)))
            else: 
                
                map_forward, varying_params_shape = kernel_args

                init_map_params, nonlinear_apply_map = hk.without_apply_rng(hk.transform(partial(map_forward, 
                                                                                out_dim=out_dim)))
                
                apply_map = lambda params, input: nonlinear_apply_map(params.amortized, params.varying, input)

                def get_random_params(key):
                    return init_map_params(key, jnp.empty((varying_params_shape,)), jnp.empty((self.in_dim,)))
                
                def format_params(params):
                    return params 

        self._apply_map = apply_map 
        self.get_random_params = get_random_params
        self._format_params = format_params
        self._noise_sample, self._noise_logpdf = noise_dist.sample, noise_dist.logpdf
        
    def map(self, state, params):
        return GaussianParams(*self._apply_map(params, state))
    
    def sample(self, key, state, params):
        return self._noise_sample(key, self.map(state, params))

    def logpdf(self, x, state, params):
        return self._noise_logpdf(x, self.map(state, params))

    def format_params(self, params):
        return self._format_params(params)

    
class HMM: 

    default_prior_base_scale = 5e-2
    default_transition_base_scale = 5e-2
    default_emission_base_scale = 2e-2

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_kernel_type, 
                emission_kernel_type,
                prior_dist=Gaussian):

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

    def __init__(self, filt_dist, backwd_kernel):

        self.filt_dist:Gaussian = filt_dist
        self.backwd_kernel:Kernel = backwd_kernel

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

    @abstractmethod
    def backwd_pass(self, last_filt_state, backwd_state_seq):
        raise NotImplementedError

    def compute_filt_seq(self, obs_seq, formatted_params):


        init_filt_state = self.init_filt_state(obs_seq[0], formatted_params)

        @jit
        def _step(carry, x):
            filt_state, params = carry
            obs = x
            filt_state = self.new_filt_state(obs, filt_state, params)
            return (filt_state, params), filt_state

        filt_state_seq = lax.scan(_step, init=(init_filt_state, formatted_params), xs=obs_seq[1:])[1]


        return tree_prepend(init_filt_state, filt_state_seq)

    def compute_backwd_seq(self, filt_seq, formatted_params):
        
        return vmap(self.new_backwd_state, in_axes=(0,None))(tree_droplast(filt_seq), formatted_params)

    def smooth_seq(self, obs_seq, params):
        
        formatted_params = self.format_params(params)

        formatted_params.compute_covs()

        filt_state_seq = self.compute_filt_seq(obs_seq, formatted_params)
        backwd_state_seq = self.compute_backwd_seq(filt_state_seq, formatted_params)

        return self.backwd_pass(tree_get_idx(-1, filt_state_seq), backwd_state_seq)

class LinearBackwardSmoother(BackwardSmoother):

    def __init__(self, state_dim, filt_dist=Gaussian):

        backwd_kernel_def = ({'homogeneous':False, 'map':'linear'}, None)

        super().__init__(filt_dist, Kernel(state_dim, state_dim, backwd_kernel_def))

    def new_backwd_state(self, filt_state, params):

        A, a, Q = *params.transition.map, params.transition.scale.cov
        mu, Sigma = filt_state.mean, filt_state.scale.cov
        I = jnp.eye(self.state_dim)

        K = Sigma @ A.T @ inv(A @ Sigma @ A.T + Q)
        C = I - K @ A

        A_back = K 
        a_back = C @ mu - K @ a
        cov_back = C @ Sigma

        return KernelParams(LinearMapParams(A_back, a_back), Scale(cov=cov_back))


    def backwd_pass(self, last_filt_state, backwd_state_seq):

        last_filt_state_mean, last_filt_state_cov = last_filt_state.mean, last_filt_state.scale.cov

        @jit
        def _step(filt_state, backwd_state):
            A_back, a_back, cov_back = *backwd_state.map, backwd_state.scale.cov
            filt_state_mean, filt_state_cov = filt_state
            mean = A_back @ filt_state_mean + a_back
            cov = A_back @ filt_state_cov @ A_back.T + cov_back
            return (mean, cov), GaussianParams(mean, Scale(cov=cov))

        marginals = lax.scan(_step, 
                                init=(last_filt_state_mean, last_filt_state_cov), 
                                xs=backwd_state_seq, 
                                reverse=True)[1]
        
        return tree_append(marginals, GaussianParams(last_filt_state_mean, Scale(cov=last_filt_state_cov)))

class LinearGaussianHMM(HMM, LinearBackwardSmoother):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning):

        transition_kernel_def = ({'homogeneous':True, 'map':'linear'}, transition_matrix_conditionning)
        emission_kernel_def =  ({'homogeneous':True, 'map':'linear'}, None)

        HMM.__init__(self, 
                    state_dim, 
                    obs_dim, 
                    transition_kernel_type = lambda state_dim:Kernel(state_dim, state_dim, transition_kernel_def), 
                    emission_kernel_type = lambda state_dim, obs_dim:Kernel(state_dim, obs_dim, emission_kernel_def))

        LinearBackwardSmoother.__init__(self, state_dim)


    def init_filt_state(self, obs, params):

        mean, cov =  Kalman.init(obs, params.prior, params.emission)

        return GaussianParams(mean, Scale(cov=cov))

    def new_filt_state(self, obs, filt_state, params):

        pred_mean, pred_cov = Kalman.predict(filt_state.mean, filt_state.scale.cov, params.transition)
        mean, cov = Kalman.update(pred_mean, pred_cov, obs, params.emission)

        return GaussianParams(mean, Scale(cov=cov))

    def likelihood_seq(self, obs_seq, params):

        return Kalman.filter_seq(obs_seq, self.format_params(params))[-1]
    
    def gaussianize_filt_state(self, filt_state, params):
        return filt_state

class NonLinearGaussianHMM(HMM):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning,
                hidden_layer_sizes,
                slope,
                num_particles=1000):

        nonlinear_map_forward = partial(neural_map, hidden_layer_sizes=hidden_layer_sizes, slope=slope)
        transition_kernel_def = ({'homogeneous':True, 'map':'linear'}, transition_matrix_conditionning)
        emission_kernel_def = ({'homogeneous':True, 'map':'nonlinear'}, nonlinear_map_forward)
        
        HMM.__init__(self, 
                    state_dim, 
                    obs_dim, 
                    transition_kernel_type = lambda state_dim: Kernel(state_dim, state_dim, transition_kernel_def), 
                    emission_kernel_type  = lambda state_dim, obs_dim:Kernel(state_dim, obs_dim, emission_kernel_def))

        self.smc = SMC(self.transition_kernel, self.emission_kernel, self.prior_dist, num_particles)

    def likelihood_seq(self, key, obs_seq, params):

        return self.smc.filter_seq(key, 
                            obs_seq, 
                            self.format_params(params))[-1]
    
    def compute_filt_seq(self, key, obs_seq, params):

        return self.smc.compute_filt_seq(key, 
                                obs_seq, 
                                self.format_params(params))
        
    def backwd_pass(self, filt_seq, params, key):

        formatted_params = self.format_params(params)

        return self.smc.smooth_from_filt_seq(key, filt_seq, formatted_params)
    
    def smooth_seq(self, key, obs_seq, params):

        key, subkey = random.split(key, 2)

        formatted_params = self.format_params(params)

        filt_seq = self.smc.compute_filt_seq(key, 
                                obs_seq, 
                                formatted_params)

        return self.smc.smooth_from_filt_seq(subkey, filt_seq, formatted_params)


class NeuralLinearBackwardSmoother(LinearBackwardSmoother):

    @staticmethod
    def filt_update_forward(obs, pred_state, hidden_layer_sizes, out_dim):
        net = hk.nets.MLP((*hidden_layer_sizes,out_dim), 
                        activation=nn.tanh,
                        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                        activate_final=False)
        return net(jnp.concatenate((obs, pred_state)))

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_kernel_matrix_conditionning='diagonal', 
                filt_update_hidden_layer_sizes = (100,), 
                prior_dist=Gaussian, 
                filt_dist=Gaussian):
        
        super().__init__(state_dim)

        self.state_dim, self.obs_dim = state_dim, obs_dim 

        self.prior_dist:Gaussian = prior_dist
        transition_kernel_def = ({'homogeneous':True, 'map':'linear'}, transition_kernel_matrix_conditionning)
        self.transition_kernel = Kernel(state_dim, state_dim, transition_kernel_def)
        self.filt_dist:Gaussian = filt_dist

        d = state_dim
        self.filt_state_shape = d + d*(d+1) // 2
        
        self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(self.filt_update_forward, 
                                                                                hidden_layer_sizes=filt_update_hidden_layer_sizes, 
                                                                                out_dim=self.filt_state_shape))) 
        
    def get_random_params(self, key):

        subkeys = random.split(key, 4)

        dummy_obs = jnp.empty((self.obs_dim,))

        prior_params = self.prior_dist.get_random_params(subkeys[0], self.state_dim, HMM.default_prior_base_scale)
        transition_params = self.transition_kernel.get_random_params(subkeys[1], HMM.default_transition_base_scale)
        filt_update_params = self.filt_update_init_params(subkeys[2], dummy_obs, jnp.empty((self.filt_state_shape,)))

        return NeuralLinearBackwardSmootherParams(prior_params, transition_params, filt_update_params)

    def init_filt_state(self, obs, params):

        pred_state = vec_from_gaussian_params(params.prior, self.state_dim)

        filt_state = self.filt_update_apply(params.filt_update, obs, pred_state)
        return gaussian_params_from_vec(filt_state, self.state_dim, chol_add=jnp.eye)

    def new_filt_state(self, obs, filt_state, params):

        pred_mean, pred_cov = Kalman.predict(filt_state.mean, filt_state.scale.cov, params.transition)
        pred_state = vec_from_gaussian_params(GaussianParams(pred_mean, Scale(cov=pred_cov)), self.state_dim)

        filt_state  = self.filt_update_apply(params.filt_update, obs, pred_state)

        return gaussian_params_from_vec(filt_state, self.state_dim, chol_add=jnp.eye)
    
    def format_params(self, params):
        return NeuralLinearBackwardSmootherParams(self.prior_dist.format_params(params.prior),
                                                self.transition_kernel.format_params(params.transition),
                                                params.filt_update)

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', sum(len(leaf) for leaf in tree_leaves(params)))
        print('-- in prior + predict + backward:', sum(len(leaf) for leaf in tree_leaves((params.prior, params.transition))))
        print('-- in update:', sum(len(leaf) for leaf in tree_leaves(params.filt_update)))

class NeuralBackwardSmoother(BackwardSmoother):


    @staticmethod
    def filt_update_forward(obs, prev_filt_state, hidden_layer_sizes, out_dim):
        net = hk.nets.MLP((*hidden_layer_sizes, out_dim), 
                        activation=nn.tanh,
                        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                        activate_final=False)

        out = net(jnp.concatenate((obs, prev_filt_state)))
        return out

    @staticmethod
    def backwd_kernel_map_forward(varying_params, next_state, hidden_layer_sizes, out_dim):

        d = out_dim
        out_dim = d + (d * (d+1)) // 2

        net = hk.nets.MLP((*hidden_layer_sizes, out_dim),
                hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                activation=nn.tanh,
                activate_final=False)
        
        out = net(jnp.concatenate((varying_params, next_state)))
        mean = out[:d]
        chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(out[d:]) + jnp.eye(d)

        return mean, Scale(chol=chol)

        
    def __init__(self, state_dim, obs_dim,
                filt_update_hidden_layer_sizes=(10,),
                backwd_map_hidden_layer_sizes=(10,),
                prior_dist=Gaussian, 
                filt_dist=Gaussian,
                backwd_dist=Gaussian):


        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.prior_dist = prior_dist 

        d = state_dim
        self.filt_state_shape = d + d*(d+1) // 2

        backwd_kernel_def = ({'homogeneous':False, 'map':'nonlinear'}, 
                        (partial(self.backwd_kernel_map_forward, hidden_layer_sizes=backwd_map_hidden_layer_sizes), self.filt_state_shape))

        super().__init__(filt_dist, Kernel(state_dim, state_dim, backwd_kernel_def, backwd_dist))


        self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(self.filt_update_forward, 
                                                                                hidden_layer_sizes=filt_update_hidden_layer_sizes, 
                                                                                out_dim=self.filt_state_shape)))

    def get_random_params(self, key):

        subkeys = random.split(key, 3)
        prior_params = self.prior_dist.get_random_params(subkeys[0], self.state_dim, HMM.default_prior_base_scale)
        backwd_map_params = self.backwd_kernel.get_random_params(subkeys[1])
        filt_update_params = self.filt_update_init_params(subkeys[2], jnp.empty((self.obs_dim,)), jnp.empty((self.filt_state_shape,)))

        return NeuralBackwardSmootherParams(prior_params, filt_update_params, backwd_map_params)

    def format_params(self, params):

        formatted_prior_params = self.prior_dist.format_params(params.prior)

        return NeuralBackwardSmootherParams(formatted_prior_params, params.filt_update, params.backwd_map)

    def init_filt_state(self, obs, params):
        filt_state = self.filt_update_apply(params.filt_update, obs, vec_from_gaussian_params(params.prior, self.state_dim))
        return gaussian_params_from_vec(filt_state, self.state_dim, chol_add=jnp.eye)

    def new_filt_state(self, obs, filt_state, params):
        filt_state = self.filt_update_apply(params.filt_update, obs, vec_from_gaussian_params(filt_state, self.state_dim))
        return gaussian_params_from_vec(self.filt_update_apply(params.filt_update, obs, filt_state), self.state_dim, chol_add=jnp.eye)
    
    def new_backwd_state(self, filt_state, params):
        return BackwardState(vec_from_gaussian_params(filt_state, self.state_dim), params.backwd_map)

    def backwd_pass(self, last_filt_state, backwd_state_seq):
        pass 

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', sum(len(leaf) for leaf in tree_leaves(params)))
        print('-- in prior:', sum(len(leaf) for leaf in tree_leaves(params.prior)))
        print('-- in filtering update:', sum(len(leaf) for leaf in tree_leaves(params.filt_update)))
        print('-- in backward map:', sum(len(leaf) for leaf in tree_leaves(params.backwd_map)))


    

        




        

            
        



        

