from .distributions import *
from .kernels import *
from abc import ABCMeta, abstractmethod
from collections import namedtuple
def set_parametrization(args):
    Scale.parametrization = args.parametrization
from jax import lax, numpy as jnp 
from jax.flatten_util import ravel_pytree

State = namedtuple('State', ['out','hidden'])
GeneralBackwdState = namedtuple('BackwardState', ['inner', 'varying'])


class BackwardSmoother(metaclass=ABCMeta):

    def __init__(self, filt_dist, backwd_kernel):

        self.filt_dist:Gaussian = filt_dist
        self.backwd_kernel:ParametricKernel = backwd_kernel


    @abstractmethod
    def smoothing_means_tm1_t(self, filt_params, backwd_params, *args):
        raise NotImplementedError
    
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
    def backwd_params_from_states(self, state, params):
        raise NotImplementedError

    @abstractmethod
    def backwd_step(self, *args):
        raise NotImplementedError
    
    @abstractmethod
    def compute_marginals(self, *args):
        raise NotImplementedError

    @abstractmethod
    def smooth_seq(self, *args):
        raise NotImplementedError

    @abstractmethod
    def log_fwd_potential(self, x_0, x_1, states, params):
        raise NotImplementedError
    
    def compute_state_seq(self, obs_seq, compute_up_to, formatted_params):

        mask_seq = jnp.arange(0, len(obs_seq)) <= compute_up_to

        init_state = self.init_state(obs_seq[0], 
                                    formatted_params)

        def false_fun(obs, prev_state):
            return prev_state
        
        def true_fun(obs, prev_state):
            return self.new_state(obs, 
                                  prev_state, 
                                  formatted_params)

        def _step(carry, x):
            prev_state = carry
            obs, mask = x
            state = lax.cond(mask, 
                             true_fun, false_fun, 
                            obs, prev_state)
            return state, state

        state_seq = lax.scan(_step, init=init_state, xs=(obs_seq[1:], mask_seq[1:]))[1]

        return tree_prepend(init_state, state_seq)

    def get_states(self, 
                  t, 
                  base_state, 
                  ys_for_bptt, 
                  formatted_params):

        bptt_depth = len(ys_for_bptt)

        timesteps = jnp.arange(0, bptt_depth) - bptt_depth + 1 

        masks_compute = (timesteps + t >= 0)
        masks_init = (timesteps + t == 0)

        def false_fun(mask_init, obs, prev_state):
            return prev_state

        def true_fun(mask_init, obs, prev_state):

            def init(prev_state):
                return self.init_state(obs, formatted_params)
            
            def update(prev_state):
                return self.new_state(obs, prev_state, formatted_params)

            return lax.cond(mask_init, init, update, prev_state)

        def _step(carry, x):
            prev_state = carry
            mask_compute, mask_init, obs = x
            state = lax.cond(mask_compute, 
                            true_fun, 
                            false_fun, 
                            mask_init, obs, prev_state)
            
            return state, state

        state_seq = lax.scan(_step, init=base_state, 
                                          xs=(masks_compute, masks_init, ys_for_bptt))[1]

        return tree_get_idx(0, state_seq), (tree_get_idx(-2, state_seq), tree_get_idx(-1, state_seq))

    def compute_filt_params_seq(self, state_seq, formatted_params):
        return vmap(self.filt_params_from_state, in_axes=(0,None))(state_seq, formatted_params)

    def compute_backwd_params_seq(self, state_seq, formatted_params):
        state_seq_strided = (tree_droplast(state_seq), tree_dropfirst(state_seq))
        return vmap(self.backwd_params_from_states, in_axes=(0,None))(state_seq_strided, formatted_params)
    
    def new_proposal_params(self, transition_params, filt_params):
        raise NotImplementedError


class TwoFilterSmoother(metaclass=ABCMeta):
        
    def __init__(self, state_dim, forward_kernel:ParametricKernel):
        self.marginal_dist = Gaussian
        self.forward_kernel:ParametricKernel = forward_kernel(state_dim, state_dim)

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


def linear_gaussian_forward_params_from_backwd_variable_and_transition(filt_params:Gaussian.Params, 
                                                                        transition_params:ParametricKernel.Params):
    A, b, Q_prec = transition_params.map.w, transition_params.map.b, transition_params.noise.scale.prec

    prec_forward = 2 * (Q_prec + filt_params.scale.prec)

    K = inv(prec_forward)

    A_forward = K @ Q_prec @ A
    b_forward = K @ (Q_prec @ b + filt_params.scale.prec @ filt_params.mean)

    return ParametricKernel.Params(map=Maps.LinearMapParams(A_forward, b_forward), 
                        noise=Gaussian.NoiseParams(Scale(prec=prec_forward)))

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

        return ParametricKernel.Params(Maps.LinearMapParams(A_back, a_back), Gaussian.NoiseParams(Scale(cov=cov_back)))

    def __init__(self, state_dim):

        backwd_kernel_def = {'map_type':'linear',
                            'map_info' : {'conditionning': None, 
                                        'bias': True,
                                        'range_params':(0,1)}}

        super().__init__(filt_dist=Gaussian, 
                        backwd_kernel=ParametricKernel(state_dim, state_dim, backwd_kernel_def))
        
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

    def backwd_step(self, current_marginal, backwd_params):
        A_back, a_back, cov_back = backwd_params.map.w, backwd_params.map.b, backwd_params.noise.scale.cov
        smoothed_mean, smoothed_cov = current_marginal
        mean = A_back @ smoothed_mean + a_back
        cov = A_back @ smoothed_cov @ A_back.T + cov_back
        return (mean, cov), Gaussian.Params(mean=mean, 
                                            scale=Scale(cov=cov))

    def compute_joint_marginals(self, filt_params_seq, backwd_params_seq, lag):
        
        def _compute_joint_marginal(filt_params, backward_params_subseq):

            lagged_filt_params_mean, lagged_filt_params_cov = filt_params.mean, filt_params.scale.cov

            @jit
            def _marginal_step(current_marginal, backwd_params):
                A_back, a_back, cov_back = backwd_params.map.w, backwd_params.map.b, backwd_params.noise.scale.cov
                smoothed_mean, smoothed_cov = current_marginal
                mean = A_back @ smoothed_mean + a_back
                cov = A_back @ smoothed_cov @ A_back.T + cov_back
                return (mean, cov), Gaussian.Params(mean=mean, scale=Scale(cov=cov))

            marginals = lax.scan(_marginal_step, 
                                    init=(lagged_filt_params_mean, lagged_filt_params_cov), 
                                    xs=backward_params_subseq, 
                                    reverse=True)[1]

            return tree_append(marginals, filt_params)

        return vmap(_compute_joint_marginal)(tree_get_slice(lag, -1, filt_params_seq), 
                                            tree_get_strides(stride=lag, tree=backwd_params_seq))

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
            return marginals.mean, marginals.scale.cov     
        else: 
            marginals = self.compute_joint_marginals(filt_params_seq, backwd_params_seq, lag)
            return marginals
       
    def new_proposal_params(self, backwd_params, filt_params):

        proposal_params = self.linear_gaussian_backwd_params_from_transition_and_filt(filt_params.mean, filt_params.scale.cov, backwd_params)
        return proposal_params

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

    def log_fwd_potential(self, x_0, x_1, params):
        return self.transition_kernel.logpdf(x_1, x_0, params.transition)
    
    def smoothing_means_tm1_t(self, filt_params, backwd_params, *args):
        mean, cov = filt_params.mean, filt_params.scale.cov
        return self.backwd_step((mean,cov), backwd_params)[0][0], filt_params.mean