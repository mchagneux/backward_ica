from .distributions import *
from .kernels import *
from abc import ABCMeta, abstractmethod
from collections import namedtuple
def set_parametrization(args):
    Scale.parametrization = args.parametrization
from jax import lax, numpy as jnp 


State = namedtuple('State', ['out','hidden'])
GeneralBackwdState = namedtuple('BackwardState', ['inner', 'varying'])


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
    def empty_state(self):
        raise NotImplementedError
        
    @abstractmethod
    def init_state(self, obs, params):
        raise NotImplementedError
    
    @abstractmethod
    def new_state(self, obs, prev_state, params):
        raise NotImplementedError

    @abstractmethod
    def filt_params_from_state(self, state, params):
        raise NotImplementedError

    @abstractmethod
    def backwd_params_from_state(self, state, params):
        raise NotImplementedError

    @abstractmethod
    def compute_marginals(self, *args):
        raise NotImplementedError

    @abstractmethod
    def smooth_seq(self, *args):
        raise NotImplementedError

    def compute_state_seq(self, obs_seq, compute_up_to, formatted_params):

        mask_seq = jnp.arange(0, len(obs_seq)) <= compute_up_to

        init_state = self.init_state(obs_seq[0], 
                                    formatted_params)

        def false_fun(obs, prev_state, params):
            return prev_state

        @jit
        def _step(carry, x):
            prev_state, params = carry
            obs, mask = x
            state = lax.cond(mask, self.new_state, false_fun, 
                            obs, prev_state, params)
            return (state, params), state

        state_seq = lax.scan(_step, init=(init_state, formatted_params), xs=(obs_seq[1:], mask_seq[1:]))[1]

        return tree_prepend(init_state, state_seq)

    def compute_filt_params_seq(self, state_seq, formatted_params):
        return vmap(self.filt_params_from_state, in_axes=(0,None))(state_seq, formatted_params)

    def compute_backwd_params_seq(self, state_seq, formatted_params):
        return vmap(self.backwd_params_from_state, in_axes=(0,None))(tree_droplast(state_seq), formatted_params)

class TwoFilterSmoother(metaclass=ABCMeta):
        
    def __init__(self, state_dim, forward_kernel:Kernel):
        self.marginal_dist = Gaussian
        self.forward_kernel:Kernel = forward_kernel(state_dim, state_dim)

    @abstractmethod
    def init_filt_params(self, state, params):
        raise NotImplementedError

    @abstractmethod
    def new_filt_params(self, state, prev_filt_params, params):
        raise NotImplementedError

    @abstractmethod
    def init_backwd_var(self, state, params):
        raise NotImplementedError

    @abstractmethod
    def new_backwd_var(self, state, next_backwd_var, params):
        raise NotImplementedError

    @abstractmethod
    def compute_filt_params_seq(self, state_seq, formatted_params):
        raise NotImplementedError

    @abstractmethod
    def compute_backwd_variables_seq(self, state_seq, compute_up_to, formatted_params):
        raise NotImplementedError

    @abstractmethod 
    def forward_params_from_backwd_var(self, backwd_var, params):
        raise NotImplementedError
    
    @abstractmethod
    def compute_marginal(self, filt_params, backwd_variable):
        raise NotImplementedError

    @abstractmethod
    def compute_marginals(self, filt_params_seq, backwd_variables_seq):
        raise NotImplementedError

    @abstractmethod
    def smooth_seq(self, obs_seq, params):
        raise NotImplementedError

    @abstractmethod
    def filt_seq(self, obs_seq, params):

        raise NotImplementedError

    @abstractmethod
    def compute_state_seq(self, obs_seq, formatted_params):
        raise NotImplementedError

class LinearBackwardSmoother(BackwardSmoother):

    @staticmethod
    def linear_gaussian_backwd_params_from_transition_and_filt(filt_mean, filt_cov, params):

        A, a, Q = params.map.w, params.map.b, params.noise.scale.cov
        mu, Sigma = filt_mean, filt_cov
        I = jnp.eye(a.shape[0])

        K = Sigma @ A.T @ inv(A @ Sigma @ A.T + Q)
        C = I - K @ A

        A_back = K 
        a_back = C @ mu - K @ a
        cov_back = C @ Sigma

        return Kernel.Params(Maps.LinearMapParams(A_back, a_back), Gaussian.NoiseParams(Scale(cov=cov_back)))

    def __init__(self, state_dim):

        backwd_kernel_def = {'map_type':'linear',
                            'map_info' : {'conditionning': None, 
                                        'bias': True,
                                        'range_params':(0,1)}}

        super().__init__(filt_dist=Gaussian, 
                        backwd_kernel=Kernel(state_dim, state_dim, backwd_kernel_def))
        
    def compute_marginals(self, last_filt_params, backwd_params_seq):
        last_filt_params_mean, last_filt_params_cov = last_filt_params.mean, last_filt_params.scale.cov

        @jit
        def _step(filt_params, backwd_params):
            A_back, a_back, cov_back = backwd_params.map.w, backwd_params.map.b, backwd_params.noise.scale.cov
            smoothed_mean, smoothed_cov = filt_params
            mean = A_back @ smoothed_mean + a_back
            cov = A_back @ smoothed_cov @ A_back.T + cov_back
            return (mean, cov), Gaussian.Params(mean=mean, scale=Scale(cov=cov))

        marginals = lax.scan(_step, 
                                init=(last_filt_params_mean, last_filt_params_cov), 
                                xs=backwd_params_seq, 
                                reverse=True)[1]
        
        marginals = tree_append(marginals, Gaussian.Params(mean=last_filt_params_mean, 
                                                        scale=Scale(cov=last_filt_params_cov)))

        return marginals

    def compute_fixed_lag_marginals(self, filt_params_seq, backwd_params_seq, lag):
        
        def _compute_fixed_lag_marginal(init, x):

            lagged_filt_params, backwd_params_subseq = x

            lagged_filt_params_mean, lagged_filt_params_cov = lagged_filt_params.mean, lagged_filt_params.scale.cov

            @jit
            def _marginal_step(current_marginal, backwd_params):
                A_back, a_back, cov_back = backwd_params.map.w, backwd_params.map.b, backwd_params.noise.scale.cov
                smoothed_mean, smoothed_cov = current_marginal
                mean = A_back @ smoothed_mean + a_back
                cov = A_back @ smoothed_cov @ A_back.T + cov_back
                return (mean, cov), None

            marginal = lax.scan(_marginal_step, 
                                    init=(lagged_filt_params_mean, lagged_filt_params_cov), 
                                    xs=backwd_params_subseq, 
                                    reverse=True)[0]

            return None, Gaussian.Params(mean=marginal[0], scale=Scale(cov=marginal[1]))

        return lax.scan(_compute_fixed_lag_marginal, 
                            init=None, 
                            xs=(tree_get_slice(lag, None, filt_params_seq), tree_get_strides(lag, backwd_params_seq)))[1]

    def filt_seq(self, obs_seq, params):
        formatted_params = self.format_params(params)

        state_seq = self.compute_state_seq(obs_seq, len(obs_seq)-1, formatted_params)
        filt_params_seq = self.compute_filt_params_seq(state_seq, formatted_params)
        return vmap(lambda x:x.mean)(filt_params_seq), vmap(lambda x:x.scale.cov)(filt_params_seq)
    
    def smooth_seq(self, obs_seq, params, lag=None):
        
        formatted_params = self.format_params(params)

        state_seq = self.compute_state_seq(obs_seq, len(obs_seq)-1, formatted_params)
        filt_params_seq = self.compute_filt_params_seq(state_seq, formatted_params)
        backwd_params_seq = self.compute_backwd_params_seq(state_seq, formatted_params)

        if lag is None: 
            marginals = self.compute_marginals(tree_get_idx(-1, filt_params_seq), backwd_params_seq)
        else: 
            marginals = self.compute_fixed_lag_marginals(filt_params_seq, backwd_params_seq, lag)

        return marginals.mean, marginals.scale.cov     

    def smooth_seq_at_multiple_timesteps(self, obs_seq, params, slices):
        formatted_params = self.format_params(params)


        state_seq = self.compute_state_seq(obs_seq, len(obs_seq)-1, formatted_params)
        filt_params_seq = self.compute_filt_params_seq(state_seq, formatted_params)
        backwd_params_seq = self.compute_backwd_params_seq(state_seq, formatted_params)


        def smooth_up_to_timestep(timestep):
            marginals = self.compute_marginals(tree_get_idx(timestep, filt_params_seq), tree_get_slice(0, timestep-1, backwd_params_seq))
            return marginals.mean, marginals.scale.cov
        means, covs = [], []

        for timestep in slices:
            mean, cov = smooth_up_to_timestep(timestep)
            means.append(mean)
            covs.append(cov)
            
        return means, covs  
