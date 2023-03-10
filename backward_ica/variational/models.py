from backward_ica.stats.distributions import * 
from backward_ica.stats.kernels import * 
from backward_ica.stats import LinearBackwardSmoother, TwoFilterSmoother, State, BackwardSmoother
from backward_ica.stats.kalman import Kalman
from backward_ica.stats.hmm import HMM

from jax.tree_util import tree_leaves
from jax import numpy as jnp, lax
from backward_ica.utils import * 
import copy
from typing import Any 
import backward_ica.variational.inference_nets as inference_nets
from collections import namedtuple
from jax.flatten_util import ravel_pytree

class NeuralBackwardSmoother(BackwardSmoother):

    @register_pytree_node_class
    @dataclass(init=True)
    class Params:

        prior:Any 
        state:Any 
        backwd:Any
        filt:Any

        def compute_covs(self):
            if hasattr(self.backwd, 'noise'):
                self.backwd.noise.scale.cov
                self.backwd.noise.scale.prec



        def tree_flatten(self):
            return ((self.prior, self.state, self.backwd, self.filt), None)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

    
    @classmethod
    def with_linear_gaussian_transition_kernel(cls, args):

        transition_kernel = Kernel.linear_gaussian(
                                        matrix_conditonning=args.transition_matrix_conditionning,
                                        bias=args.transition_bias, 
                                        range_params=args.range_transition_map_params)(
                                                                args.state_dim, 
                                                                args.state_dim)
                                                    
        return cls(
                args.state_dim, 
                args.obs_dim, 
                transition_kernel, 
                args.backwd_layers,
                args.update_layers)

    def __init__(self, 
            state_dim,
            obs_dim, 
            transition_kernel:Kernel=None,
            backwd_layers=(8,8),
            update_layers=(8,8)):
        

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.transition_kernel:Kernel = transition_kernel
        self.update_layers = update_layers
        self.backwd_layers = backwd_layers
        d = self.state_dim

        if self.transition_kernel is not None: 

            backwd_kernel_def = {'map_type':'linear',
                                'map_info' : {'conditionning': None, 
                                        'bias': True,
                                        'range_params':(0,1)}}
            
            super().__init__(
                        filt_dist=Gaussian, 
                        backwd_kernel=Kernel(
                                        state_dim, 
                                        state_dim, 
                                        backwd_kernel_def))
            
            def backwd_params_from_state(state, params):
                filt_params = self.filt_params_from_state(state, params)
                return LinearBackwardSmoother.linear_gaussian_backwd_params_from_transition_and_filt(
                                                                    filt_params.mean, 
                                                                    filt_params.scale.cov, 
                                                                    params.backwd), filt_params

            self._backwd_params_from_state = backwd_params_from_state

            self._log_transition_function = lambda params, x_0, x_1: \
                        self.transition_kernel.logpdf(x_1, x_0, params)
            
            


        else: 
            net = lambda aux, x_1, state_dim: inference_nets.johnson(
                                                aux,
                                                x_1, 
                                                layers=backwd_layers, 
                                                state_dim=state_dim)
            
            net = lambda aux, input, state_dim: inference_nets.johnson(aux, input, backwd_layers, state_dim)
            
            def _backwd_map(aux, input, state_dim):
                eta1_filt, eta2_filt = aux.eta1, aux.eta2
                out = net(aux, input, state_dim)
                eta1_backwd, eta2_backwd = out[0] + eta1_filt, out[1] + eta2_filt
                out_params = Gaussian.Params(eta1=eta1_backwd, eta2=eta2_backwd)
                return (out_params.mean, out_params.scale)
            
            def _log_transition_function(x_0, x_1, aux):

                eta1, eta2, const = net(aux, x_1, state_dim)
                
                # eta2 = out[1]
                # eta1 = out[0] #- 2 * mu_filt @ eta2.T


                return x_0.T @ eta2 @ x_0 \
                    + eta1.T @ x_0 + const
            
            def backwd_params_from_state(state, params):
                filt_params = self.filt_params_from_state(state, params)
                return params.backwd, filt_params
            
            self._log_transition_function = hk.without_apply_rng(
                                                hk.transform(
                                                _log_transition_function))[1]
                
            self._backwd_params_from_state = backwd_params_from_state

            backwd_kernel_def = {
                            'map_type':'nonlinear',
                            'map_info' : {
                                        'homogeneous': False, 
                                        'dummy_varying_params':
                                                Gaussian.Params(
                                                    eta1=jnp.empty((self.state_dim,)),
                                                    eta2=jnp.eye(self.state_dim)) 
                                                },
                            'map': _backwd_map}

            super().__init__(
                        filt_dist=Gaussian,
                        backwd_kernel=Kernel(state_dim, state_dim, backwd_kernel_def))


        
        self._state_net = hk.without_apply_rng(hk.transform(
                                                partial(inference_nets.deep_gru, 
                                                        layers=self.update_layers)))
        
        self._filt_net = hk.without_apply_rng(hk.transform(
                                                partial(inference_nets.gaussian_proj, 
                                                        d=d)))


    def log_transition_function(self, x_0, x_1, params):
        return self._log_transition_function(params[0], x_0, x_1, params[1])
            
    def compute_marginals(self, *args):
        return super().compute_marginals(*args)
    
    def smooth_seq(self, *args):
        return super().smooth_seq(*args)
    
    def get_random_params(self, key, params_to_set=None):

        key_prior, key_state, key_filt, key_backwd = random.split(key, 4)

        dummy_obs = jnp.empty((self.obs_dim,))


        prior_params = tuple([random.normal(key, shape=[size]) for \
                                                            key, size in zip(
                                                    random.split(key_prior, len(self.update_layers)), 
                                                    self.update_layers)])

        state_params = self._state_net.init(key_state, dummy_obs, prior_params)

        out, new_state = self._state_net.apply(state_params, dummy_obs, prior_params)

        dummy_state = State(out=out, 
                            hidden=new_state)

        filt_params = self._filt_net.init(key_filt, dummy_state)

        if self.transition_kernel is None:
            backwd_params = self.backwd_kernel.get_random_params(key_backwd)
        else: 
            backwd_params = self.transition_kernel.get_random_params(key_backwd)


        params =  self.Params(prior_params, 
                            state_params, 
                            backwd_params,
                            filt_params)
        
        if params_to_set is not None:
            params = self.set_params(params, params_to_set)
        return params  
        

    def frozen_prior(self):
        return tuple([jnp.zeros(shape=[size]) for size in self.update_layers])

    def set_params(self, params, args):
        new_params = copy.deepcopy(params)
        for k,v in vars(args).items():         
            if (k == 'default_transition_base_scale') and (self.transition_kernel is not None): 
                new_params.backwd.noise.scale = Scale.set_default(
                                                params.backwd.noise.scale, 
                                                v, 
                                                Scale.parametrization)
       
        return new_params

    def format_params(self, params):

        if self.transition_kernel is None:
            return params 
        else: 
            return self.Params(params.prior, 
                                params.state, 
                                self.transition_kernel.format_params(params.backwd), 
                                params.filt)

    def init_state(self, obs, params):
        out, init_state = self._state_net.apply(params.state, obs, params.prior)
        return State(out=out, hidden=init_state)

    def new_state(self, obs, prev_state, params):
        out, new_state = self._state_net.apply(params.state, obs, prev_state.hidden)
        return State(out=out, hidden=new_state)

    def filt_params_from_state(self, state, params):
        return self._filt_net.apply(params.filt, state)

    def backwd_params_from_state(self, state, params):
        return self._backwd_params_from_state(state, params)

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', len(ravel_pytree(params)[0]))
    
        
    def empty_state(self):
        params = self.format_params(self.get_random_params(jax.random.PRNGKey(0)))
        return self.init_state(jnp.empty((self.obs_dim,)), params)

@register_pytree_node_class
@dataclass(init=True)
class JohnsonParams:

    prior: Gaussian.Params
    transition:Kernel.Params
    net:Any

    def compute_covs(self):
        self.prior.scale.cov
        self.transition.noise.scale.cov

    def tree_flatten(self):
        return ((self.prior, self.transition, self.net), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

class JohnsonSmoother:


    def __init__(self, 
                    state_dim, 
                    obs_dim, 
                    transition_matrix_conditionning,
                    range_transition_map_params,
                    transition_bias,
                    layers, 
                    anisotropic):

        self.state_dim = state_dim 
        self.obs_dim = obs_dim 
        self.prior_dist = Gaussian

        self.transition_kernel = Kernel.linear_gaussian(
                                            matrix_conditonning=transition_matrix_conditionning,
                                            bias=transition_bias, 
                                            range_params=range_transition_map_params)(
                                                                        state_dim, 
                                                                        state_dim)

        net = inference_nets.johnson_anisotropic if anisotropic else inference_nets.johnson
        self._net = hk.without_apply_rng(hk.transform(partial(net, layers=layers, state_dim=state_dim)))

    def get_random_params(self, key, params_to_set=None):
        key_prior, key_transition, key_net = random.split(key, 3)                                       

        prior_params = self.prior_dist.get_random_params(key_prior, self.state_dim)
        transition_params = self.transition_kernel.get_random_params(key_transition)
        net_params = self._net.init(key_net, jnp.empty((self.obs_dim,)))

        params = JohnsonParams(prior_params, transition_params, net_params)
        if params_to_set is not None: 
            params = self.set_params(params, params_to_set)
        return params

    def format_params(self, params):
        return JohnsonParams(self.prior_dist.format_params(params.prior), 
                            self.transition_kernel.format_params(params.transition),
                            params.net)

    def set_params(self, params, args):
        new_params = copy.deepcopy(params)
        for k,v in vars(args).items():         
            if k == 'default_transition_base_scale': 
                new_params.transition.noise.scale = Scale.set_default(
                                                                    params.transition.noise.scale, 
                                                                    v, 
                                                                    Scale.parametrization)
            elif k == 'default_prior_base_scale':
                new_params.prior.scale = Scale.set_default(params.prior.scale, 
                                                           v, 
                                                           Scale.parametrization)
        return new_params

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', len(ravel_pytree(params)[0]))
    
    def log_transition_function(self, x_0, x_1, params):
        return self.transition_kernel.logpdf(x_1, x_0, params.transition)

class JohnsonBackward(JohnsonSmoother, LinearBackwardSmoother):


    @classmethod
    def from_args(cls, args):
        return cls(
            args.state_dim, 
            args.obs_dim, 
            args.transition_matrix_conditionning,
            args.range_transition_map_params, 
            args.transition_bias, 
            args.update_layers, 
            args.anisotropic)


    def __init__(
            self, 
            state_dim,
            obs_dim, 
            transition_matrix_conditionning, 
            range_transition_map_params, 
            transition_bias, 
            update_layers, 
            anisotropic):

        JohnsonSmoother.__init__(
                            self, 
                            state_dim, 
                            obs_dim, 
                            transition_matrix_conditionning, 
                            range_transition_map_params, 
                            transition_bias, 
                            update_layers, 
                            anisotropic)
        
        LinearBackwardSmoother.__init__(self, state_dim)


    def init_state(self, obs, params):
        out = self._net.apply(params.net, obs)
        return Gaussian.Params.from_nat_params(out[0] + params.prior.eta1, out[1] + params.prior.eta2)

    def new_state(self, obs, prev_state, params):

        pred_mean, pred_cov = Kalman.predict(prev_state.mean, prev_state.scale.cov, params.transition)  

        pred = Gaussian.Params.from_mean_cov(pred_mean, pred_cov)
        out = self._net.apply(params.net, obs)

        return Gaussian.Params.from_nat_params(out[0] + pred.eta1, out[1] + pred.eta2)

    def filt_params_from_state(self, state, params):
        return state

    def backwd_params_from_state(self, state, params):
        return self.linear_gaussian_backwd_params_from_transition_and_filt(
                                                            state.mean, 
                                                            state.scale.cov, 
                                                            params.transition)

    def compute_state_seq(self, obs_seq, compute_up_to, formatted_params):
        formatted_params.compute_covs()
        return super().compute_state_seq(obs_seq, compute_up_to, formatted_params)

    def empty_state(self):
        eta1 = jnp.empty((self.state_dim,))
        eta2 = jnp.empty((self.state_dim, self.state_dim))
        return Gaussian.Params.from_nat_params(eta1, eta2)
        



#     def __init__(self, 
#                 state_dim, 
#                 obs_dim,
#                 update_layers,
#                 backwd_layers,
#                 filt_dist=Gaussian):

#         self.state_dim = state_dim 
#         self.obs_dim = obs_dim 

#         self.update_layers = update_layers

#         self.filt_params_shape = jnp.sum(jnp.array(update_layers))

#         self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(self.filt_update_forward, 
#                                                                 layers=update_layers, 
#                                                                 state_dim=state_dim)))
        

#         backwd_kernel_map_def = {'map_type':'nonlinear',
#                                 'map_info' : {'homogeneous': False, 'varying_params_shape':self.filt_params_shape},
#                                 'map': partial(self.backwd_update_forward, layers=backwd_layers)}
                

#         super().__init__(filt_dist, 
#                         Kernel(state_dim, state_dim, backwd_kernel_map_def, Gaussian))

#     def get_random_params(self, key, args=None):
        
#         key_prior, key_filt, key_back = random.split(key, 3)
    
#         dummy_obs = jnp.ones((self.obs_dim,))


#         key_priors = random.split(key_prior, len(self.update_layers))
#         prior_params = tuple([random.normal(key, shape=[size]) for key, size in zip(key_priors, self.update_layers)])
#         filt_update_params = self.filt_update_init_params(key_filt, dummy_obs, prior_params)

#         backwd_params = self.backwd_kernel.get_random_params(key_back)

#         return GeneralBackwardSmootherParams(prior=prior_params,
#                                             filt_update=filt_update_params, 
#                                             backwd=backwd_params)

#     def smooth_seq(self, key, obs_seq, params, num_samples, lag=None):

#         formatted_params = self.format_params(params)

#         filt_params_seq = self.compute_filt_params_seq(obs_seq, formatted_params)
#         backwd_params_seq = self.compute_backwd_params_seq(filt_params_seq, formatted_params)

#         if lag is None:
#             marginals = self.compute_marginals(key, filt_params_seq, backwd_params_seq, num_samples)
#         else: 
#             marginals = self.compute_fixed_lag_marginals(key, filt_params_seq, backwd_params_seq, num_samples, lag)

#         return jnp.mean(marginals, axis=0), jnp.var(marginals, axis=0)

#     def format_params(self, params):
#         return params

#     def init_filt_params(self, obs, params):
#         return FiltState(*self.filt_update_apply(params.filt_update, obs, params.prior))

#     def new_filt_params(self, obs, filt_params:FiltState, params):
#         return FiltState(*self.filt_update_apply(params.filt_update, obs, filt_params.hidden))

#     def get_init_state(self):
#         return tuple([jnp.zeros(shape=[size]) for size in self.update_layers])

#     def new_backwd_params(self, filt_params:FiltState, params):

#         return BackwardState(params.backwd, jnp.concatenate(filt_params.hidden))

#     def compute_marginals(self, key, filt_params_seq, backwd_params_seq, num_samples):

#         def _sample_for_marginals(key, last_filt_params:FiltState, backwd_params_seq):
            
#             keys = random.split(key, backwd_params_seq.varying.shape[0]+1)

#             last_sample = self.filt_dist.sample(keys[-1], last_filt_params.out)

#             def _sample_step(next_sample, x):
                
#                 key, backwd_params = x
#                 sample = self.backwd_kernel.sample(key, next_sample, backwd_params)
#                 return sample, sample
            
#             samples = lax.scan(_sample_step, init=last_sample, xs=(keys[:-1], backwd_params_seq), reverse=True)[1]

#             return tree_append(samples, last_sample)

#         parallel_sampler = jit(vmap(_sample_for_marginals, in_axes=(0,None,None)))

#         return parallel_sampler(random.split(key, num_samples), tree_get_idx(-1, filt_params_seq), backwd_params_seq)

#     def compute_fixed_lag_marginals(self, key, filt_params_seq, backwd_params_seq, num_samples, lag):
        
#         def _sample_for_marginals(key, filt_params_seq, backwd_params_seq):

#             def _sample_for_marginal(init, x):

#                 key, lagged_filt_params, strided_backwd_params_subseq = x

#                 keys = random.split(key, strided_backwd_params_subseq.varying.shape[0]+1)

#                 last_sample = self.filt_dist.sample(keys[-1], lagged_filt_params.out)

#                 def _sample_step(next_sample, x):
                    
#                     key, backwd_params = x
#                     sample = self.backwd_kernel.sample(key, next_sample, backwd_params)
#                     return sample, None

#                 marginal_sample = lax.scan(_sample_step, 
#                                         init=last_sample, 
#                                         xs=(keys[:-1], strided_backwd_params_subseq), 
#                                         reverse=True)[0]

#                 return None, marginal_sample
                

#             return lax.scan(_sample_for_marginal, 
#                                 init=None, 
#                                 xs=(random.split(key, backwd_params_seq.varying.shape[0]-lag+1), tree_get_slice(lag, None, filt_params_seq), tree_get_strides(lag, backwd_params_seq)))[1]
        
#         parallel_sampler = jit(vmap(_sample_for_marginals, in_axes=(0,None,None)))

#         return parallel_sampler(random.split(key, num_samples), filt_params_seq, backwd_params_seq)
    
#     def print_num_params(self):
#         params = self.get_random_params(random.PRNGKey(0))
#         print('Num params:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params)))
#         print('-- filt net:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.filt_update))))
#         print('-- prior state:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.prior))))
#         print('-- backwd net:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.backwd))))
        


            
        



        

