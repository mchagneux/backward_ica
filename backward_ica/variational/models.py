from backward_ica.stats.distributions import * 
from backward_ica.stats.kernels import * 
from backward_ica.stats import LinearBackwardSmoother, TwoFilterSmoother, State
from backward_ica.stats.kalman import Kalman
from backward_ica.stats.hmm import HMM

from jax.tree_util import tree_leaves
from jax import numpy as jnp, lax
from backward_ica.utils import * 
import copy
from typing import Any 
import backward_ica.variational.inference_nets as inference_nets
from collections import namedtuple



class NeuralLinearBackwardSmoother(LinearBackwardSmoother):

    @register_pytree_node_class
    @dataclass(init=True)
    class Params:

        prior:Any 
        state:Any 
        backwd:Any
        filt:Any

        def compute_covs(self):
            self.prior.scale.cov
            self.backwd.noise.scale.cov

        def tree_flatten(self):
            return ((self.prior, self.state, self.backwd, self.filt), None)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

    @classmethod
    def with_transition_from_p(cls, p:HMM, layers=(8,8)):
        return cls(p.state_dim, 
                    p.obs_dim, 
                    p.transition_kernel, 
                    layers)
    
    @classmethod
    def with_linear_gaussian_transition_kernel(cls, p:HMM, layers):

        transition_kernel = Kernel.linear_gaussian(matrix_conditonning='init_sym_def_pos', 
                                                        bias=True, 
                                                        range_params=(-1,1))(p.state_dim, p.state_dim)
                                                        
        return cls(p.state_dim, p.obs_dim, transition_kernel, layers)


    def __init__(self, 
                state_dim,
                obs_dim, 
                transition_kernel=None,
                update_layers=(8,8)):
        

        super().__init__(state_dim)

        self.state_dim = state_dim
        self.obs_dim = obs_dim

        self.transition_kernel:Kernel = transition_kernel
        self.update_layers = update_layers
        d = self.state_dim
        
        self._state_net = hk.without_apply_rng(hk.transform(partial(inference_nets.deep_gru, 
                                                                    layers=self.update_layers)))
        self._filt_net = hk.without_apply_rng(hk.transform(partial(inference_nets.gaussian_proj, 
                                                                    d=d)))


        if self.transition_kernel is None:
            self._backwd_net = hk.without_apply_rng(hk.transform(partial(inference_nets.linear_gaussian_proj, d=d)))
            self._backwd_params_from_state = lambda state, params: self._backwd_net.apply(params.backwd, state)

        else: 
            def backwd_params_from_state(state, params):
                filt_params = self.filt_params_from_state(state, params)
                return NeuralLinearBackwardSmoother.linear_gaussian_backwd_params_from_transition_and_filt(filt_params.mean, 
                                                                                                        filt_params.scale.cov, 
                                                                                                        params.backwd)

            self._backwd_params_from_state = backwd_params_from_state
             
    def get_random_params(self, key, params_to_set=None):

        key_prior, key_state, key_filt, key_backwd = random.split(key, 4)

        dummy_obs = jnp.empty((self.obs_dim,))


        prior_params = tuple([random.normal(key, shape=[size]) for key, size in zip(random.split(key_prior, len(self.update_layers)), 
                                                                                    self.update_layers)])

        state_params = self._state_net.init(key_state, dummy_obs, prior_params)

        out, new_state = self._state_net.apply(state_params, dummy_obs, prior_params)

        dummy_state = State(out=out, 
                            hidden=new_state)

        filt_params = self._filt_net.init(key_filt, dummy_state)

        if self.transition_kernel is None:
            backwd_params = self._backwd_net.init(key_backwd, dummy_state)
        else: 
            backwd_params = self.transition_kernel.get_random_params(key_backwd)


        params =  self.Params(prior_params, 
                            state_params, 
                            backwd_params,
                            filt_params)
        
        if params_to_set is not None:
            params = self.set_params(params, params_to_set)
        return params  
        
    def set_params(self, params, args):
        new_params = copy.deepcopy(params)
        for k,v in vars(args).items():         
            if (k == 'default_transition_base_scale') and (self.transition_kernel is not None): 
                new_params.backwd.noise.scale = Scale.set_default(params.backwd.noise.scale, v, Scale.parametrization)
       
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
        print('Num params:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params)))
        print('-- in prior:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.prior,))))
        print('-- in state:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.state,))))
        print('-- in backwd:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.backwd,))))
        print('-- in filt:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params.filt)))
    

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

    def __init__(self, state_dim, obs_dim, layers, anisotropic):

        self.state_dim = state_dim 
        self.obs_dim = obs_dim 
        self.prior_dist = Gaussian

        self.transition_kernel = Kernel.linear_gaussian(matrix_conditonning='diagonal',
                                                        bias=False, 
                                                        range_params=(-1,1))(state_dim, state_dim)

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
                new_params.transition.noise.scale = Scale.set_default(params.transition.noise.scale, v, Scale.parametrization)
            elif k == 'default_prior_base_scale':
                new_params.prior.scale = Scale.set_default(params.prior.scale, v, Scale.parametrization)
        return new_params

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params)))
        print('-- in prior:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.prior,))))
        print('-- in transition:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.transition,))))
        print('-- in net:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.net,))))
    
class JohnsonBackward(JohnsonSmoother, LinearBackwardSmoother):

    def __init__(self, state_dim, obs_dim, layers, anisotropic):

        JohnsonSmoother.__init__(self, state_dim, obs_dim, layers, anisotropic)
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
        return self.linear_gaussian_backwd_params_from_transition_and_filt(state.mean, state.scale.cov, params.transition)

    def compute_state_seq(self, obs_seq, compute_up_to, formatted_params):
        formatted_params.compute_covs()
        return super().compute_state_seq(obs_seq, compute_up_to, formatted_params)

BackwdVar = namedtuple('BackwdVar', ['base', 'tilde'])

class JohnsonForward(JohnsonSmoother, TwoFilterSmoother):
    
    @staticmethod
    def linear_gaussian_forward_params_from_backwd_variable_and_transition(backwd_variable_tilde:Gaussian.Params, 
                                                                            transition_params:Kernel.Params):
        A, R_prec = transition_params.map.w, transition_params.noise.scale.prec

        eta1, eta2 = backwd_variable_tilde.eta1, backwd_variable_tilde.eta2
        prec_forward = R_prec + eta2

        K = inv(prec_forward)

        A_forward = K @ R_prec @ A
        b_forward = K @ eta1

        return Kernel.Params(map=Maps.LinearMapParams(A_forward, b_forward), 
                            noise=Gaussian.NoiseParams(Scale(prec=prec_forward)))
        
    def __init__(self, state_dim, obs_dim, layers, anisotropic):
        JohnsonSmoother.__init__(self, state_dim, obs_dim, layers, anisotropic)
        
        TwoFilterSmoother.__init__(self, state_dim, 
                                    forward_kernel=Kernel.linear_gaussian(matrix_conditonning=None, 
                                                                        bias=True, 
                                                                        range_params=(0,1)))

    def init_filt_params(self, state, params):
        return Gaussian.Params.from_nat_params(state[0] + params.prior.eta1, state[1] + params.prior.eta2)

    def new_filt_params(self, state, prev_filt_params, params):
        pred_mean, pred_cov = Kalman.predict(prev_filt_params.mean, prev_filt_params.scale.cov, params.transition)  

        pred = Gaussian.Params.from_mean_cov(pred_mean, pred_cov)

        return Gaussian.Params.from_nat_params(state[0] + pred.eta1, state[1] + pred.eta2)
        
    def init_backwd_var(self, state, params):


        d = self.state_dim 
        base = Gaussian.Params.from_nat_params(eta1=jnp.zeros((d,)), 
                                               eta2=jnp.zeros((d,d)))               


        return BackwdVar(base=base, tilde=Gaussian.Params.from_nat_params(*state))

    def compute_state(self, obs, params):
        return self._net.apply(params.net, obs)

    def new_backwd_var(self, state, next_backwd_var, params):

        next_eta1_tilde, next_eta2_tilde = next_backwd_var.tilde.eta1, next_backwd_var.tilde.eta2

        A, R = params.transition.map.w, params.transition.noise.scale.cov
        K = inv(jnp.eye(self.state_dim) + next_eta2_tilde @ R)

        base = Gaussian.Params.from_nat_params(eta1 = A.T @ K @ next_eta1_tilde, 
                                                eta2 = A.T @ K @ next_eta2_tilde @ A)


        tilde = Gaussian.Params.from_nat_params(state[0] + base.eta1, 
                                                state[1] + base.eta2)


        return BackwdVar(base=base,
                        tilde=tilde)

    def forward_params_from_backwd_var(self, backwd_var:BackwdVar, params):
        return self.linear_gaussian_forward_params_from_backwd_variable_and_transition(backwd_var.tilde, params.transition)

    def compute_state_seq(self, obs_seq, formatted_params):
        formatted_params.compute_covs()
        return vmap(self.compute_state, in_axes=(0,None))(obs_seq, formatted_params)

    def compute_filt_params_seq(self, state_seq, formatted_params):

        init_filt_params = self.init_filt_params(tree_get_idx(0,state_seq), 
                                                formatted_params)

        @jit
        def _step(carry, state):
            prev_filt_params, formatted_params = carry
            filt_params = self.new_filt_params(state, prev_filt_params, formatted_params)
            return (filt_params, formatted_params), filt_params

        filt_params_seq = lax.scan(_step, 
                            init=(init_filt_params, formatted_params), 
                            xs=tree_dropfirst(state_seq))[1]

        return tree_prepend(init_filt_params, filt_params_seq)

    def compute_backwd_variables_seq(self, state_seq, compute_up_to, formatted_params):

        empty_backwd_var_comp = Gaussian.Params.from_nat_params(eta1=jnp.empty((self.state_dim,)), 
                                        eta2=jnp.empty((self.state_dim,self.state_dim)))      
        empty_backwd_var = BackwdVar(base=empty_backwd_var_comp, tilde=empty_backwd_var_comp)

        @jit
        def _step(carry, x):

            next_backwd_var, params = carry 
            idx = x

            def false_fun(idx, next_backwd_var, params):

                return empty_backwd_var

            def true_fun(idx, next_backwd_var, params):

                def last_term(idx, next_backwd_var, params):
                    return self.init_backwd_var(tree_get_idx(idx, state_seq), params)
                def other_terms(idx, next_backwd_var, params):
                    return self.new_backwd_var(tree_get_idx(idx, state_seq), next_backwd_var, params)

                return lax.cond(idx < compute_up_to, other_terms, last_term, idx, next_backwd_var, params)

            backwd_var = lax.cond(idx <= compute_up_to, true_fun, false_fun, idx, next_backwd_var, params)

            return (backwd_var, params), backwd_var
        

        backwd_variables_seq = lax.scan(_step, 
                                        init=(empty_backwd_var, formatted_params),
                                        xs=jnp.arange(0, len(state_seq[0])),
                                        reverse=True)[1]

        return backwd_variables_seq

    def compute_marginal(self, filt_params:Gaussian.Params, backwd_variable:Gaussian.Params):
        mu, Sigma = filt_params.mean, filt_params.scale.cov
        kappa, Pi = backwd_variable.base.eta1, backwd_variable.base.eta2
        K = Sigma @ inv(jnp.eye(self.state_dim) + Pi @ Sigma)
        marginal_mean = mu + K @ (kappa - Pi @ mu)
        marginal_cov = Sigma - K @ Pi @ Sigma
        return Gaussian.Params(mean=marginal_mean, scale=Scale(cov=marginal_cov))

    def compute_marginals(self, filt_params_seq, backwd_variables_seq):
        
        return vmap(self.compute_marginal)(filt_params_seq, backwd_variables_seq)

    def smooth_seq(self, obs_seq, params, lag=None):
        
        formatted_params = self.format_params(params)
        formatted_params.compute_covs()

        state_seq = self.compute_state_seq(obs_seq, formatted_params)
        marginal_smoothing_stats =  self.compute_marginals(self.compute_filt_params_seq(state_seq, formatted_params),
                                                            self.compute_backwd_variables_seq(state_seq, len(obs_seq)-1, formatted_params))

        return marginal_smoothing_stats.mean, marginal_smoothing_stats.scale.cov

    def smooth_seq_at_multiple_timesteps(self, obs_seq, params, slices):
        formatted_params = self.format_params(params)
        formatted_params.compute_covs()
        state_seq = self.compute_state_seq(obs_seq, formatted_params)
        filt_params_seq = self.compute_filt_params_seq(state_seq, formatted_params)


        def smooth_up_to_timestep(timestep):
            marginals = self.compute_marginals(filt_params_seq=tree_get_slice(0, timestep, filt_params_seq), 
                                                backwd_variables_seq=self.compute_backwd_variables_seq(tree_get_slice(0, timestep, state_seq), timestep-1, formatted_params))
            return marginals.mean, marginals.scale.cov
        means, covs = [], []

        for timestep in slices:
            mean, cov = smooth_up_to_timestep(timestep)
            means.append(mean)
            covs.append(cov)
            
        return means, covs  

    def filt_seq(self, obs_seq, params):
        formatted_params = self.format_params(params)
        formatted_params.compute_covs()
        
        state_seq = self.compute_state_seq(obs_seq, formatted_params)
        filt_params_seq =  self.compute_filt_params_seq(state_seq, formatted_params)

        return vmap(lambda x:x.mean)(filt_params_seq), vmap(lambda x:x.scale.cov)(filt_params_seq)
        





# class NeuralBackwardSmoother(BackwardSmoother):

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
        


            
        



        

