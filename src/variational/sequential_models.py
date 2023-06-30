from src.stats.distributions import * 
from src.stats.kernels import * 
from src.stats import LinearBackwardSmoother, State, BackwardSmoother
from src.stats.kalman import Kalman
from src.stats.hmm import HMM

from jax.tree_util import tree_leaves
from jax import numpy as jnp, lax
from src.utils.misc import * 
import copy



from typing import Any 
import src.variational.inference_nets as inference_nets
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


        def tree_flatten(self):
            return ((self.prior, self.state, self.backwd, self.filt), None)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

    @classmethod
    def with_linear_gaussian_transition_kernel(cls, args):

        transition_kernel = ParametricKernel.linear_gaussian(
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
            transition_kernel:ParametricKernel=None,
            backwd_layers=(8,8),
            update_layers=(8,8),
            conjugate=True):
        

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.transition_kernel:ParametricKernel = transition_kernel
        self.update_layers = update_layers
        self.backwd_layers = backwd_layers
        d = self.state_dim
        



        if self.transition_kernel is not None: 

            print('Setting up potentials from linear Gaussian kernel.')
            backwd_kernel_def = {'map_type':'linear',
                            'map_info' : {'conditionning': None, 
                                        'bias': True,
                                        'range_params':(0,1)}}
            
            
            def backwd_params_from_filt_params(filt_params_0, filt_params_1, params):
                mean_filt_0, cov_filt_0 = filt_params_0.mean, filt_params_0.scale.cov
                

                return LinearBackwardSmoother.linear_gaussian_backwd_params_from_transition_and_filt(mean_filt_0, 
                                                                                                     cov_filt_0, 
                                                                                                     params.backwd)
                                                                        


            def _log_fwd_potential(x_0, x_1, params):
                return self.transition_kernel.logpdf(x_1, x_0, params.transition)
            

        elif conjugate: 
            
            def _backwd_map(aux, x_1, state_dim):
                filt_params_0 = aux[0]
                filt_params_1 = aux[1]
                eta1_filt, eta2_filt = aux[0].eta1, aux[0].eta2
                eta1_potential, eta2_potential = inference_nets.backwd_net(filt_params_0.vec, x_1, backwd_layers, state_dim)
                eta1_backwd, eta2_backwd = eta1_filt + eta1_potential, eta2_filt + eta2_potential 
                out_params = Gaussian.Params(eta1=eta1_backwd, eta2=eta2_backwd)


                # eta1_filt, eta2_filt = aux[0].eta1, aux[0].eta2
                # mu_0 = aux[0].mean
                # mu_1 = aux[1].mean
                # eta1_potential, eta2_potential = inference_nets.backwd_net(aux[0].vec, x_1-mu_1, backwd_layers, state_dim)
                # # eta1_backwd, eta2_backwd = eta1_potential + eta1_filt, eta2_potential + eta2_filt
                # eta1_backwd, eta2_backwd = eta1_filt + eta1_potential - 2 * eta2_potential.T @ mu_0, eta2_filt + eta2_potential 
                # out_params = Gaussian.Params(eta1=eta1_backwd, eta2=eta2_backwd)

                return (out_params.mean, out_params.scale), (eta1_potential, eta2_potential)
                            

            dummy_gaussian_params = Gaussian.Params(eta1=jnp.empty((self.state_dim,)),
                                                    eta2=jnp.eye(self.state_dim))
            backwd_kernel_def = {
                            'map_type':'nonlinear',
                            'map_info' : {
                                        'homogeneous': False, 
                                        'dummy_varying_params':
                                                (dummy_gaussian_params, dummy_gaussian_params)
                                                },
                            'map': _backwd_map}
            

            def _log_fwd_potential(x_0, x_1, backwd_params):

                params, aux = backwd_params

                _ , (eta1, eta2) = self.backwd_kernel.nonlinear_map_apply(params, aux, x_1)
                
                return x_0.T @ eta2 @ x_0 + eta1.T @ x_0
            
            def backwd_params_from_filt_params(filt_params_0, filt_params_1, params):
                return params.backwd, (filt_params_0, filt_params_1)
                        


        super().__init__(
                filt_dist=Gaussian,
                backwd_kernel=ParametricKernel(state_dim, state_dim, backwd_kernel_def))
            
        self._log_fwd_potential =  _log_fwd_potential
            
        self._backwd_params_from_filt_params = backwd_params_from_filt_params
            
        self._state_net = hk.without_apply_rng(hk.transform(
                                                partial(inference_nets.deep_gru, 
                                                        layers=self.update_layers)))
        
        self._filt_net = hk.without_apply_rng(hk.transform(
                                                partial(inference_nets.gaussian_proj, 
                                                        d=d)))

    def backwd_step(self, key, current_sample, backwd_params):
        return self.backwd_kernel.sample(key, current_sample, backwd_params)
    
    def log_fwd_potential(self, x_0, x_1, backwd_params):
        return self._log_fwd_potential(x_0, x_1, backwd_params)
            
    def compute_marginals(self, *args):
        return super().compute_marginals(*args)
    
    def smooth_seq(self, key, obs_seq, params, num_samples):
        params = self.format_params(params)
        T = len(obs_seq) - 1 
        state_seq = self.compute_state_seq(obs_seq, 
                                        compute_up_to=T , 
                                        formatted_params=params)
        
        timesteps = jnp.arange(T+1)

        def _step(carry, x):
            t, state_0, key = x 
            sample, state_1 = carry
            def _last_step(state_0, sample):
                filt_params = self.filt_params_from_state(state_0, params)
                return self.filt_dist.sample(key, filt_params) 
            def _backwd_step(state_0, sample):
                backwd_params = self.backwd_params_from_states((state_0, state_1), params)
                return self.backwd_kernel.sample(key, sample, backwd_params)
            
            new_sample = lax.cond(t<T,
                                  _backwd_step,
                                  _last_step,
                                  state_0, sample)
            return (new_sample, state_0), new_sample


        sampler = lambda key: jax.lax.scan(_step, 
                               init=(jnp.empty((self.state_dim,)), tree_get_idx(-1, state_seq)),
                               xs=(timesteps, state_seq, jax.random.split(key, T+1)))[1]
        
        samples = jax.vmap(sampler)(jax.random.split(key, num_samples))

        return jnp.mean(samples, axis=0), jnp.var(samples, axis=0)
    
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
            formatted_transition = self.transition_kernel.format_params(params.backwd)
            formatted_transition.noise.scale = Scale(cov=jnp.eye(self.state_dim))
            # formatted_transition.map.w = jnp.eye(self.state_dim)
            return self.Params(params.prior, 
                                params.state, 
                                formatted_transition, 
                                params.filt)
        
    def smoothing_means_tm1_t(self, filt_params, backwd_params, num_samples, key):
        key_t, key_tm1 = jax.random.split(key, 2)
        samples_t = jax.vmap(self.filt_dist.sample, in_axes=(0,None))(jax.random.split(key_t, 
                                                                                       num_samples), 
                                                                    filt_params)
        samples_tm1 = jax.vmap(self.backwd_kernel.sample, in_axes=(0,0,None))(jax.random.split(key_tm1, num_samples), 
                                                                              samples_t, 
                                                                              backwd_params)
        
        return jnp.mean(samples_tm1, axis=0), filt_params.mean
    
    def init_state(self, obs, params):
        out, init_state = self._state_net.apply(params.state, obs, params.prior)
        return State(out=out, hidden=init_state)

    def new_state(self, obs, prev_state, params):
        out, new_state = self._state_net.apply(params.state, obs, prev_state.hidden)
        return State(out=out, hidden=new_state)

    def filt_params_from_state(self, state, params):
        return self._filt_net.apply(params.filt, state)

    def backwd_params_from_states(self, states, params):
        filt_params_0 = self.filt_params_from_state(states[0], params)
        filt_params_1 = self.filt_params_from_state(states[1], params)
        return self._backwd_params_from_filt_params(filt_params_0, filt_params_1, params)

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', len(ravel_pytree(params)[0]))
        
    def empty_state(self):
        params = self.format_params(self.get_random_params(jax.random.PRNGKey(0)))
        return self.init_state(jnp.empty((self.obs_dim,)), params)

class NonAmortizedBackwardSmoother(BackwardSmoother):

    @register_pytree_node_class
    @dataclass(init=True)
    class Params:
        backwd:Any
        filt:Any

        def tree_flatten(self):
            return ((self.backwd, self.filt), None)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)
        
    def __init__(self, state_dim, obs_dim, backwd_layers):
        
        self.state_dim = state_dim
        self.obs_dim = obs_dim

        
        # def _backwd_map(aux, x_1, state_dim):
        #     filt_params_0 = aux[0]
        #     filt_params_1 = aux[1]
        #     eta1_filt, eta2_filt = aux[0].eta1, aux[0].eta2
        #     eta1_potential, eta2_potential = inference_nets.backwd_net(filt_params_0.vec, x_1, backwd_layers, state_dim)
        #     eta1_backwd, eta2_backwd = eta1_filt + eta1_potential, eta2_filt + eta2_potential 
        #     out_params = Gaussian.Params(eta1=eta1_backwd, 
        #                                  eta2=eta2_backwd)


        #     # eta1_filt, eta2_filt = aux[0].eta1, aux[0].eta2
        #     # mu_0 = aux[0].mean
        #     # mu_1 = aux[1].mean
        #     # eta1_potential, eta2_potential = inference_nets.backwd_net(aux[0].vec, x_1-mu_1, backwd_layers, state_dim)
        #     # # eta1_backwd, eta2_backwd = eta1_potential + eta1_filt, eta2_potential + eta2_filt
        #     # eta1_backwd, eta2_backwd = eta1_filt + eta1_potential - 2 * eta2_potential.T @ mu_0, eta2_filt + eta2_potential 
        #     # out_params = Gaussian.Params(eta1=eta1_backwd, eta2=eta2_backwd)

        #     return (out_params.mean, out_params.scale), (eta1_potential, eta2_potential)
        
        def _backwd_map(aux, x_1, state_dim):
            eta1_filt, eta2_filt = aux[0].eta1, aux[0].eta2
            eta1_potential = inference_nets.nonamortized_backwd_net_for_mean(x_1, backwd_layers, state_dim)

            log_eta_2_param = hk.get_parameter('backwd_sigma_diag', 
                                               shape=(state_dim,), 
                                               init=jnp.zeros)
            
            eta2_potential = -jnp.diag(jnp.exp(log_eta_2_param)**2)
            eta1_backwd, eta2_backwd = eta1_filt + eta1_potential, eta2_filt + eta2_potential
            out_params = Gaussian.Params(eta1=eta1_backwd, eta2=eta2_backwd)

            return (out_params.mean, out_params.scale), (eta1_potential, eta2_potential)
                        

        # def _backwd_map(aux, x_1, state_dim):
        #     # eta1_filt, eta2_filt = aux[0].eta1, aux[0].eta2
        #     mu = inference_nets.nonamortized_backwd_net_for_mean(x_1, backwd_layers, state_dim)

        #     initializer = lambda shape, dtype: jnp.zeros(shape, dtype)
        #     log_std_backwd = hk.get_parameter('backwd_sigma_diag', 
        #                                        shape=(state_dim,), 
        #                                        init=initializer)
            
        #     cov_chol = jnp.diag(jnp.exp(log_std_backwd)**2)

        #     out_params = Gaussian.Params(mean=mu, scale=Scale(cov_chol=cov_chol))



        #     return (out_params.mean, out_params.scale), (None, None)
        
        
        dummy_gaussian_params = Gaussian.Params(eta1=jnp.empty((self.state_dim,)),
                                                eta2=jnp.eye(self.state_dim))
        backwd_kernel_def = {
                        'map_type':'nonlinear',
                        'map_info' : {
                                    'homogeneous': False, 
                                    'dummy_varying_params':
                                            (dummy_gaussian_params, dummy_gaussian_params)
                                            },
                        'map': _backwd_map}
        

        def _log_fwd_potential(x_0, x_1, backwd_params):

            params, aux = backwd_params

            _ , (eta1, eta2) = self.backwd_kernel.nonlinear_map_apply(params, aux, x_1)
            
            return x_0.T @ eta2 @ x_0 + eta1.T @ x_0
        
        def backwd_params_from_filt_params(filt_params_0, filt_params_1, params):
            return params.backwd, (filt_params_0, filt_params_1)

        super().__init__(
                filt_dist=Gaussian,
                backwd_kernel=ParametricKernel(state_dim, state_dim, backwd_kernel_def))
            
        self._log_fwd_potential =  _log_fwd_potential
            
        self._backwd_params_from_filt_params = backwd_params_from_filt_params

    def backwd_params_from_states(self, states, params):
        filt_params_0 = self.filt_params_from_state(states[0], params)
        filt_params_1 = self.filt_params_from_state(states[1], params)
        return self._backwd_params_from_filt_params(filt_params_0, filt_params_1, params)
    
    def backwd_step(self, key, current_sample, backwd_params):
        return self.backwd_kernel.sample(key, current_sample, backwd_params)
    
    def log_fwd_potential(self, x_0, x_1, backwd_params):
        return self._log_fwd_potential(x_0, x_1, backwd_params)
    
    def filt_params_from_state(self, state, params):
        return state
    
    def init_state(self, obs, params):
        return params.filt
    
    def new_state(self, obs, prev_state, params):
        return params.filt
    
    def empty_state(self):
        mu = jnp.empty((self.state_dim,))
        Sigma = jnp.empty((self.state_dim, self.state_dim))
                    
        return Gaussian.Params(mean=mu, scale=Scale(cov=Sigma))
    
    def smoothing_means_tm1_t(self, filt_params, backwd_params, num_samples, key):
        key_t, key_tm1 = jax.random.split(key, 2)
        samples_t = jax.vmap(self.filt_dist.sample, in_axes=(0,None))(jax.random.split(key_t, 
                                                                                        num_samples), 
                                                                    filt_params)
        samples_tm1 = jax.vmap(self.backwd_kernel.sample, in_axes=(0,0,None))(jax.random.split(key_tm1, num_samples), 
                                                                                samples_t, 
                                                                                backwd_params)
        
        return jnp.mean(samples_tm1, axis=0), filt_params.mean
    
    def get_random_params(self, key, params_to_set=None):
        key_filt, key_backwd = jax.random.split(key, 2)
        backwd_params = self.backwd_kernel.get_random_params(key_backwd)
        filt_params = self.filt_dist.get_random_params(key_filt, self.state_dim)
        return NonAmortizedBackwardSmoother.Params(backwd=backwd_params, 
                                                   filt=filt_params)
    
    def format_params(self, params):
        return NonAmortizedBackwardSmoother.Params(backwd=params.backwd, 
                                                   filt=self.filt_dist.format_params(params.filt))

    def smooth_seq(self, *args):
        pass 

    def compute_marginals(self, *args):
        pass 

    def get_states(self, t, base_state, ys_for_bptt, formatted_params):
        new_state = formatted_params.filt
        return new_state, (base_state, new_state)
    
    def print_num_params(self):
        return print('Num params:', len(ravel_pytree(self.get_random_params(jax.random.PRNGKey(0)))[0]))

@register_pytree_node_class
@dataclass(init=True)
class JohnsonParams:

    prior: Gaussian.Params
    transition:ParametricKernel.Params
    net:Any


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
        self.transition_kernel = ParametricKernel.linear_gaussian(
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
                new_params.prior._scale = Scale.set_default(params.prior.scale, 
                                                           v, 
                                                           Scale.parametrization)
        return new_params

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', len(ravel_pytree(params)[0]))
    
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
    
    @classmethod
    def from_p(cls, p:HMM, args):
        obj = cls.from_args(args)
        obj.prior_dist = p.prior_dist
        obj.transition_kernel = p.transition_kernel
        return obj

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
        
        LinearBackwardSmoother.__init__(self, 
                                        state_dim)
        
    def init_state(self, obs, params):
        out = self._net.apply(params.net, obs)
        return Gaussian.Params(eta1=out[0] + params.prior.eta1, eta2=out[1] + params.prior.eta2)

    def new_state(self, obs, prev_state, params, **kwargs):

        pred_mean, pred_cov = Kalman.predict(prev_state.mean, prev_state.scale.cov, params.transition)  

        pred = Gaussian.Params(mean=pred_mean, scale=Scale(cov=pred_cov))
        out = self._net.apply(params.net, obs)

        return Gaussian.Params(eta1=out[0] + pred.eta1, eta2=out[1] + pred.eta2)

    def filt_params_from_state(self, state, params):
        return state

    def backwd_params_from_states(self, states, params):
        state = states[0]
        return self.linear_gaussian_backwd_params_from_transition_and_filt(
                                                            state.mean, 
                                                            state.scale.cov, 
                                                            params.transition)

    def compute_state_seq(self, obs_seq, compute_up_to, formatted_params):
        return super().compute_state_seq(obs_seq, compute_up_to, formatted_params)

    def empty_state(self):
        eta1 = jnp.empty((self.state_dim,))
        eta2 = jnp.empty((self.state_dim, self.state_dim))
        return Gaussian.Params(eta1=eta1, eta2=eta2)






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
#                         ParametricKernel(state_dim, state_dim, backwd_kernel_map_def, Gaussian))

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
        


            
        



        

