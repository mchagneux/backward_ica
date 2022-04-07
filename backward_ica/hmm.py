from abc import ABCMeta, abstractmethod
from collections import namedtuple
from jax import numpy as jnp, random
import copy 
from backward_ica.kalman import kalman_init, kalman_predict, kalman_smooth_seq, kalman_update
import haiku as hk
from jax import lax, vmap, config
from functools import partial
config.update('jax_enable_x64', True)

GaussianKernelBaseParams = namedtuple('GaussianKernelParams', ['map_params', 'cov_base'])
GaussianKernelParams = namedtuple('GaussianKernelParams', ['map_params', 'cov_base', 'cov', 'prec', 'det'])

LinearGaussianKernelBaseParams = namedtuple('LinearGaussianKernelParams',['matrix', 'bias', 'cov_base'])
LinearGaussianKernelParams = namedtuple('LinearGaussianKernelParams',['matrix', 'bias', 'cov_base', 'cov', 'prec', 'det'])

GaussianBaseParams = namedtuple('GaussianParams', ['mean', 'cov_base'])
GaussianParams = namedtuple('GaussianParams', ['mean', 'cov_base', 'cov', 'prec', 'det'])

HMMParams = namedtuple('HMMParams',['prior','transition','emission'])

_conditionnings = {'diagonal':lambda param: jnp.diag(param),
                'symetric_def_pos': lambda param: param @ param.T}

def cov_prec_and_det_from_cov_chol(cov_chol):
    cov = _conditionnings['symetric_def_pos'](cov_chol)
    prec = jnp.linalg.inv(cov)
    det = jnp.linalg.det(cov)
    return cov, prec, det

def prec_and_det_from_cov(cov):
    prec = jnp.linalg.inv(cov)
    det = jnp.linalg.det(cov)
    return prec, det

class GaussianHMM(metaclass=ABCMeta): 

    default_prior_cov_base = 5e-2
    default_transition_cov_base = 5e-2
    default_emission_cov_base = 8e-3


    def __init__(self, 
                state_dim, 
                obs_dim):

        self.state_dim, self.obs_dim = state_dim, obs_dim

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
            matrix = random.uniform(subkeys[0], shape=(self.state_dim,self.state_dim))

        transition_params = LinearGaussianKernelBaseParams(matrix=matrix,
                                    bias=random.uniform(subkeys[1], shape=(self.state_dim,)),
                                    cov_base=GaussianHMM.default_transition_cov_base * jnp.ones((self.state_dim,)))

        return prior_params, transition_params

    def format_prior_and_linear_transition_params(self, prior_params, transition_params):

        prior_cov_chol = jnp.diag(prior_params.cov_base)
        prior_params = GaussianParams(prior_params.mean, 
                                    prior_cov_chol,
                                    *cov_prec_and_det_from_cov_chol(prior_cov_chol))

        transition_cov_chol = jnp.diag(transition_params.cov_base)
        transition_params = LinearGaussianKernelParams(_conditionnings[self.transition_matrix_conditionning](transition_params.matrix),
                                                    transition_params.bias,
                                                    transition_cov_chol,
                                                    *cov_prec_and_det_from_cov_chol(transition_cov_chol))
        
        return prior_params, transition_params

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

class LinearGaussianHMM(GaussianHMM, Smoother):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditioning):

        GaussianHMM.__init__(self, state_dim, obs_dim)
        Smoother.__init__(self)
        self.transition_matrix_conditionning = transition_matrix_conditioning

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
    
        prior_params, transition_params = self.format_prior_and_linear_transition_params(params.prior, params.transition)

        emission_cov_chol = jnp.diag(params.emission.cov_base)
        emission_params = LinearGaussianKernelParams(params.emission.matrix,
                                                    params.emission.bias,
                                                    emission_cov_chol,
                                                    *cov_prec_and_det_from_cov_chol(emission_cov_chol))
                                                    
        return HMMParams(prior=prior_params, transition=transition_params, emission=emission_params)

    def init_filt_state(self, obs, pred_state, params):
        mean, cov =  kalman_init(obs, params.prior, params.emission)
        return GaussianParams(mean, None, cov, *prec_and_det_from_cov(cov))

    def new_filt_state(self, obs, filt_state, params):
        pred_mean, pred_cov = kalman_predict(filt_state.mean, filt_state.cov, params.transition)
        mean, cov = kalman_update(pred_mean, pred_cov, obs, params.emission)
        return GaussianParams(mean, None, cov, *prec_and_det_from_cov(cov))

    def new_backwd_state(self, filt_state, params):

        transition_params = params.transition
        prec = transition_params.matrix.T @ transition_params.prec @ transition_params.matrix + filt_state.prec
        cov = jnp.linalg.inv(prec)

        common_term = transition_params.matrix.T @ transition_params.prec
        A = cov @ common_term
        a = cov @ (filt_state.prec @ filt_state.mean - common_term @  transition_params.bias)
        
        return LinearGaussianKernelParams(A, a, None, cov, prec, jnp.linalg.det(cov))
    
    def smooth_seq(self, obs_seq, params):
        return kalman_smooth_seq(obs_seq, params)

    # #--- debugging
    # def smooth_seq(self, obs_seq, params):
    #     filt_state = self.init_filt_state(obs_seq[0], None, params)

    #     def _forward_pass(carry, x):
    #         filt_state, params = carry 
    #         obs = x 
    #         backwd_state = self.new_backwd_state(filt_state, params)
    #         filt_state = self.new_filt_state(obs, filt_state, params)
    #         return (filt_state, params), backwd_state
    #     (last_filt_state, params), backwd_state_seq = lax.scan(_forward_pass, init=(filt_state, params), xs=obs_seq[1:])

    #     return self.backwd_pass(last_filt_state, backwd_state_seq)
    # #--- debugging


class NonLinearGaussianHMM(GaussianHMM):


    def __init__(self, state_dim, obs_dim, transition_matrix_conditionning):
        super().__init__(state_dim, obs_dim)

        def emission_map_forward(state):
            net = hk.nets.MLP((8,obs_dim,))
            return net(state)

        self.emission_map_init_params, self.emission_map_apply = hk.without_apply_rng(hk.transform(emission_map_forward))
        self.transition_matrix_conditionning = transition_matrix_conditionning

    def emission_map(self, state, params):
        return self.emission_map_apply(state, params=params.emission.map_params)

    def get_random_params(self, key):

        key, subkey = random.split(key, 2)
        prior_params, transition_params = self.init_prior_and_linear_transition(key)

        subkeys = random.split(subkey, 2)
        emission_map_params = self.emission_map_init_params(subkeys[0], jnp.empty((self.state_dim,)))
        emission_params = GaussianKernelParams(map_params=emission_map_params,
                                            cov_base=GaussianHMM.default_emission_cov_base * jnp.ones((self.obs_dim)))


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
    
    




        

