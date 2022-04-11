from abc import ABCMeta, abstractmethod
from .utils import *
from jax import numpy as jnp, random
from backward_ica.kalman import kalman_init, kalman_predict, kalman_smooth_seq, kalman_update, kalman_filter_seq
import haiku as hk
from jax import lax, vmap, config
config.update('jax_enable_x64', True)
from .utils import *


def symetric_def_pos(param):
    tril_mat = jnp.zeros()
    return param @ param.T


_conditionnings = {'diagonal':lambda param: jnp.diag(param),
                'symetric_def_pos': lambda param: param @ param.T}

class GaussianHMM(metaclass=ABCMeta): 

    default_prior_cov_base = 5e-2
    default_transition_cov_base = 5e-2
    default_emission_cov_base = 2e-2

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning):

        self.state_dim, self.obs_dim = state_dim, obs_dim
        self.transition_matrix_conditionning = transition_matrix_conditionning

    @abstractmethod
    def get_random_params(self, key):
        raise NotImplementedError

    @abstractmethod
    def format_params(self, params):
        raise NotImplementedError

    @abstractmethod
    def emission_map(self, state, params):
        raise NotImplementedError

    def transition_map(self, prev_state, params):
        return params.transition.matrix @ prev_state + params.transition.bias 

    def sample_seq(self, key, params, seq_length):
        params = self.format_params(params)
        keys = random.split(key, 2*seq_length)
        state_keys = keys[:seq_length]
        obs_keys = keys[seq_length:]

        prior_sample = random.multivariate_normal(state_keys[0], mean=params.prior.mean, cov=params.prior.cov)

        def _state_sample(carry, x):
            prev_sample = carry
            key = x
            sample = random.multivariate_normal(key, mean=self.transition_map(prev_sample, params), cov=params.transition.cov)
            return sample, sample
        _, state_seq = lax.scan(_state_sample, init=prior_sample, xs=state_keys[1:])
        state_seq = jnp.concatenate((prior_sample[None,:], state_seq))

        def _obs_sample(state_sample, key):
            return random.multivariate_normal(key, mean=self.emission_map(state_sample, params), cov=params.emission.cov)
        obs_seq = vmap(_obs_sample)(state_seq, obs_keys)

        return state_seq, obs_seq

    def init_prior_and_linear_transition(self, key):

        key, *subkeys = random.split(key, 3)
        prior_params = GaussianBaseParams(mean=random.uniform(subkeys[0], shape=(self.state_dim,)), 
                                    cov_base=GaussianHMM.default_prior_cov_base * jnp.ones((self.state_dim,)))

        subkeys = random.split(key, 3)
        if self.transition_matrix_conditionning == 'diagonal':
            matrix = random.uniform(subkeys[0], shape=(self.state_dim,))
        else: 
            d = self.state_dim
            num_free_params = (d * (d+1)) // 2 
            matrix = random.uniform(subkeys[0], shape=(num_free_params,))


        transition_params = LinearGaussianKernelBaseParams(matrix=matrix,
                                    bias=random.uniform(subkeys[1], shape=(self.state_dim,)),
                                    cov_base=GaussianHMM.default_transition_cov_base * jnp.ones((self.state_dim,)))

        return prior_params, transition_params

    def format_prior_and_linear_transition_params(self, prior_params, transition_params):

        formatted_prior_params = GaussianParams(prior_params.mean, 
                                                *cov_params_from_cov_chol(jnp.diag(prior_params.cov_base)))

        formatted_transition_params = LinearGaussianKernelParams(_conditionnings[self.transition_matrix_conditionning](transition_params.matrix),
                                                                transition_params.bias, 
                                                                *cov_params_from_cov_chol(jnp.diag(transition_params.cov_base)))
        
        return formatted_prior_params, formatted_transition_params

class Smoother(metaclass=ABCMeta):

    def __init__(self):
        pass

    @abstractmethod
    def get_random_params(self, key):
        raise NotImplementedError

    @abstractmethod
    def init_filt_state(self, obs, pred_state, params):
        raise NotImplementedError

    @abstractmethod
    def new_filt_state(self, obs, filt_state, params):
        raise NotImplementedError

    @abstractmethod
    def new_backwd_state(self, filt_state, params):
        raise NotImplementedError

    @abstractmethod
    def format_params(self, params):
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

    @abstractmethod
    def smooth_seq(self, obs_seq, params):
        raise NotImplementedError

class LinearGaussianHMM(GaussianHMM, Smoother):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditioning):

        GaussianHMM.__init__(self, state_dim, obs_dim, transition_matrix_conditioning)
        Smoother.__init__(self)

    def emission_map(self, state, params):
        return params.emission.matrix @ state + params.emission.bias 
        
    def get_random_params(self, key):
        key, subkey = random.split(key, 2)
        prior_params, transition_params = self.init_prior_and_linear_transition(key)
        
        subkeys = random.split(subkey, 3)
        emission_params = LinearGaussianKernelBaseParams(matrix=random.uniform(subkeys[0], shape=(self.obs_dim, self.state_dim)),
                                    bias=random.uniform(subkeys[1], shape=(self.obs_dim,)),
                                    cov_base=GaussianHMM.default_emission_cov_base * jnp.ones((self.obs_dim,)))

        return HMMParams(prior=prior_params, transition=transition_params, emission=emission_params)

    def format_params(self, params):
    
        formatted_prior_params, formatted_transition_params = self.format_prior_and_linear_transition_params(params.prior, params.transition)
        formatted_emission_params = LinearGaussianKernelParams(params.emission.matrix,
                                                    params.emission.bias,
                                                    *cov_params_from_cov_chol(jnp.diag(params.emission.cov_base)))
                                                    
        return HMMParams(prior=formatted_prior_params, transition=formatted_transition_params, emission=formatted_emission_params)

    def init_filt_state(self, obs, pred_state, params):

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

        # transition_params = params.transition
        # prec = transition_params.matrix.T @ transition_params.prec @ transition_params.matrix + filt_state.prec
        # cov = jnp.linalg.inv(prec)

        # common_term = transition_params.matrix.T @ transition_params.prec
        # A_back = cov @ common_term
        # a_back = cov @ (filt_state.prec @ filt_state.mean - common_term @  transition_params.bias)
        # cov_back = cov


        return LinearGaussianKernelParams(A_back, a_back, *cov_params_from_cov(cov_back))

    def likelihood_seq(self, obs_seq, params):
        return kalman_filter_seq(obs_seq, self.format_params(params))[-1]

    def smooth_seq(self, obs_seq, params):
        formatted_params = self.format_params(params)
        filt_state = self.init_filt_state(obs_seq[0], None, formatted_params)

        def _forward_pass(carry, x):
            filt_state, params = carry 
            obs = x 
            backwd_state = self.new_backwd_state(filt_state, params)
            filt_state = self.new_filt_state(obs, filt_state, params)
            return (filt_state, params), backwd_state

        (last_filt_state, _), backwd_state_seq = lax.scan(_forward_pass, init=(filt_state, formatted_params), xs=obs_seq[1:])

        return self.backwd_pass(last_filt_state, backwd_state_seq)
    
    # def smooth_seq(self, obs_seq, params):
    #     return kalman_smooth_seq(obs_seq, self.format_params(params))

# def backwd_update(shared_param, filt_mean, filt_cov):
#     net = hk.nets.MLP((8, d**2 + d + d*(d+1) // 2))
#     out = net(jnp.concatenate((shared_param, filt_mean, jnp.tril(filt_cov).flatten())))
#     A = out[:d**2].reshape((d,d))
#     a = out[d**2:d**2+d]
#     cov = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(out[d**2+d:])
#     return A, a, cov @ cov.T

# def filt_predict(shared_param, filt_mean, filt_cov):

#     net = hk.nets.MLP((8, d + d*(d+1) // 2))
#     out = net(jnp.concatenate((shared_param, filt_mean, jnp.tril(filt_cov).flatten())))
#     mean = out[:d]
#     cov_chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(out[d:])
#     return mean, cov_chol @ cov_chol.T

# def filt_update(obs, pred_mean, pred_cov):

#     net = hk.nets.MLP((8, d + d*(d+1) // 2))
#     out = net(jnp.concatenate((obs, pred_mean, jnp.tril(pred_cov).flatten())))
#     mean = out[:d]
#     cov_chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(out[d:])
#     return mean, cov_chol @ cov_chol.T

    



class NonLinearGaussianHMM(GaussianHMM):

    def __init__(self, state_dim, obs_dim, transition_matrix_conditionning):
        super().__init__(state_dim, obs_dim)

        def emission_map_forward(state):
            net = hk.nets.MLP((8,self.obs_dim))
            return net(state)

        self.emission_map_init_params, self.emission_map_apply = hk.without_apply_rng(hk.transform(emission_map_forward))
        self.transition_matrix_conditionning = transition_matrix_conditionning

    def emission_map(self, state, params):
        return self.emission_map_apply(state, params=params.emission.map_params)

    def get_random_params(self, key):
        
        key, subkey = random.split(key, 2)
        prior_params, transition_params = self.init_prior_and_linear_transition(key)
        
        subkeys = random.split(subkey, 2)
        emission_params = GaussianKernelBaseParams(map_params=self.emission_map_init_params(jnp.empty((self.state_dim,))),
                                                cov_base=GaussianHMM.default_emission_cov_base * jnp.ones((self.obs_dim,)))

        return HMMParams(prior=prior_params, transition=transition_params, emission=emission_params)

    def format_params(self, params):
        raise NotImplementedError

class NeuralSmoother(Smoother):

    def __init__(self, state_dim, obs_dim):
        self.state_dim, self.obs_dim = state_dim, obs_dim 
        
    def get_random_params(self, key):
        raise NotImplementedError

    def init_filt_state(self, obs, params):

        return self.model['filt_init'](obs=obs, 
                                    params=params['filt_init'])

    def update_filt_state(self, obs, filt_state, params):
        pred_state = self.model['filt_predict'](filt_state=filt_state, 
                                                params=params['shared'])
        filt_state = self.model['filt_update'](obs=obs,
                                            pred_state=pred_state,
                                            params=params['filt_update'])
        return filt_state
    
    def update_backwd_state(self, filt_state, params):
        return self.model['backward_update'](filt_state=filt_state,
                                            params=params['shared'])