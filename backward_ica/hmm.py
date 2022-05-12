from abc import ABCMeta, abstractmethod
from jax import numpy as jnp, random, value_and_grad, tree_util, grad
from jax.tree_util import tree_leaves
from optax import Updates
from backward_ica.kalman import Kalman
from backward_ica.smc import SMC
import haiku as hk
from jax import lax, vmap
from .utils import *
from jax.scipy.stats.multivariate_normal import logpdf as gaussian_logpdf, pdf as gaussian_pdf
from functools import partial
from jax import nn

import optax

_conditionnings = {'diagonal':lambda param: jnp.diag(param),
                'symetric_def_pos': lambda param: param @ param.T,
                None:lambda x:x}

def vec_from_linear_gaussian_kernel_params(params, d):
    return jnp.concatenate((params.map.w.flatten(), params.scale.chol[jnp.tril_indices(d)]))

def xtanh(slope):
    return lambda x: jnp.tanh(x) + slope*x

def neural_map(input, layers, slope, out_dim):

    net = hk.nets.MLP((*layers, out_dim), 
                    with_bias=False, 
                    activate_final=True, 
                    activation=xtanh(slope))

    return net(input)

def linear_map_apply(map_params, input):
    out =  jnp.dot(map_params.w, input)
    return out + jnp.broadcast_to(map_params.b, out.shape)


def linear_map_init_params(key, dummy_in, out_dim, conditionning, bias):

    if conditionning == 'diagonal':
        w = random.uniform(key, (out_dim,), minval=-1, maxval=1)
    else: 
        w = random.uniform(key, (out_dim, len(dummy_in)))
    
    return LinearMapParams(w=w, b=HMM.default_transition_bias * jnp.ones((out_dim,)))
    # return LinearMapParams(w=w)

def linear_map_format_params(params, conditionning_func):

    w = conditionning_func(params.w)
    
    if not hasattr(params, 'b'):
        b = jnp.zeros((w.shape[0],))
    else: 
        b = params.b

    return LinearMapParams(w,b)


class Gaussian: 

    @staticmethod
    def sample(key, params):
        return params.mean + params.scale.chol @ random.normal(key, (params.mean.shape[0],))
    
    @staticmethod
    def logpdf(x, params):
        return gaussian_logpdf(x, params.mean, params.scale.cov)
    
    @staticmethod
    def pdf(x, params):
        return gaussian_pdf(x, params.mean, params.scale.cov)

    @staticmethod
    def get_random_params(key, dim, default_mean=1.0, default_base_scale=None):
        
        subkeys = random.split(key,2)

        if default_mean is not None:
            mean = default_mean * jnp.ones((dim,))
        else: mean = random.uniform(subkeys[0], shape=(dim,), minval=-1, maxval=1) 
        
        if default_base_scale is not None: 
            scale = default_base_scale * jnp.ones((dim,))
        else: 
            scale = random.uniform(subkeys[1], shape=(dim,), minval=-1, maxval=1)

        return GaussianParams(mean=mean, scale=scale)

    @staticmethod
    def format_params(params):
        return GaussianParams(mean=params.mean, scale=Scale(chol=jnp.diag(params.scale)))

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
            conditionning, bias = kernel_args

            apply_map = lambda params, input: (linear_map_apply(params.map, input), params.scale)
            init_map_params = partial(linear_map_init_params, out_dim=out_dim, conditionning=conditionning, bias=bias)
            format_map_params = partial(linear_map_format_params, conditionning_func=_conditionnings[conditionning])

            def get_random_params(key, default_base_scale=None):
                key, subkey = random.split(key, 2)
                map_params = init_map_params(key, jnp.empty((self.in_dim,)))

                if default_base_scale is not None: 
                    scale=default_base_scale * jnp.ones((self.out_dim,))

                else: scale = jnp.random.uniform(subkey, shape=(self.out_dim,), minval=0.01, maxval=1)

                return KernelParams(map=map_params, scale=scale)

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
                
                def get_random_params(key, default_base_scale=None):
                    key, subkey = random.split(key, 2)
                    map_params = init_map_params(key, jnp.empty((self.in_dim,)))

                    if default_base_scale is not None: 
                        scale=default_base_scale * jnp.ones((self.out_dim,))

                    else: scale = jnp.random.uniform(subkey, shape=(self.out_dim,), minval=0.01, maxval=1)

                    return KernelParams(map=map_params, scale=scale)

                def format_params(params):
                    return KernelParams(map=format_map_params(params.map),
                                scale=Scale(chol=jnp.diag(params.scale)))

            else: 
                
                map_forward, varying_params_shape = kernel_args

                init_map_params, nonlinear_apply_map = hk.without_apply_rng(hk.transform(partial(map_forward, 
                                                                                out_dim=out_dim)))
                
                apply_map = lambda params, input: nonlinear_apply_map(params.amortized, params.varying,  input)

                def get_random_params(key):
                    return init_map_params(key, jnp.empty((varying_params_shape,)), jnp.empty((self.in_dim,)))
                
                def format_params(params):
                    return params 

        self._apply_map = apply_map 
        self.get_random_params = get_random_params
        self._format_params = format_params
        self.noise_dist:Gaussian = noise_dist
        
    def map(self, state, params):
        mean, scale = self._apply_map(params, state)
        return GaussianParams(mean=mean, scale=scale)
    
    def sample(self, key, state, params):
        return self.noise_dist.sample(key, self.map(state, params))

    def logpdf(self, x, state, params):
        return self.noise_dist.logpdf(x, self.map(state, params))
    
    def pdf(self, x, state, params):
        return self.noise_dist.pdf(x, self.map(state, params))

    def format_params(self, params):
        return self._format_params(params)
   
class HMM: 

    default_prior_base_scale = 1e-1
    default_transition_base_scale = 1e-1
    default_emission_base_scale = 1e-2
    default_transition_bias = 0.5

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
        prior_params = self.prior_dist.get_random_params(key_prior, self.state_dim, default_mean=0, default_base_scale=self.default_prior_base_scale)
        transition_params = self.transition_kernel.get_random_params(key_transition, default_base_scale=self.default_transition_base_scale)
        emission_params = self.emission_kernel.get_random_params(key_emission, default_base_scale=self.default_emission_base_scale)
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

class Smoother(metaclass=ABCMeta):

    class UpdateNet(hk.Module):
    
        def __init__(self, layers, out_dim):
            super().__init__(None)
            self.update_net = hk.nets.MLP((*layers,out_dim), 
                            activation=nn.tanh,
                            w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                            activate_final=False,
                            name='update_net')

            self.forget_net = hk.nets.MLP((1,), 
                            activation=nn.sigmoid,
                            b_init=hk.initializers.Constant(1),
                            activate_final=False,
                            name='forget_net')

        def __call__(self, obs, pred_state:GaussianParams):

            input = jnp.concatenate((obs, pred_state))

            candidate_filt_state = self.update_net(input)
            
            forget_state = self.forget_net(input)

            return candidate_filt_state #* (1 - forget_state) + pred_state * forget_state

    @staticmethod
    def filt_update_forward(obs, pred_state, layers, out_dim):

        d = pred_state.mean.shape[0]

        net = Smoother.UpdateNet(layers, out_dim)
        out = net(obs, pred_state.vec)

        return GaussianParams.from_vec(out, d)

    @staticmethod
    def backwd_kernel_map_forward(varying_params, next_state, layers, out_dim):

        d = out_dim
        out_dim = d + (d * (d+1)) // 2

        net = hk.nets.MLP((*layers, out_dim),
                w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                activation=nn.tanh,
                activate_final=False)
        
        out = net(jnp.concatenate((varying_params, next_state)))
        mean = out[:d]
        chol = jnp.zeros((d,d)).at[jnp.tril_indices(d)].set(out[d:])

        return mean, Scale(chol=chol)


    def __init__(self, filt_dist, kernel):

        self.filt_dist:Gaussian = filt_dist
        self.kernel:Kernel = kernel

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
    def new_kernel_state(self, filt_state, params):
        raise NotImplementedError

    @abstractmethod
    def compute_marginals(self, last_filt_state, backwd_state_seq):
        raise NotImplementedError

    def compute_filt_state_seq(self, obs_seq, formatted_params):
        formatted_params.compute_covs()

        init_filt_state = self.init_filt_state(obs_seq[0], formatted_params)

        @jit
        def _step(carry, x):
            filt_state, params = carry
            obs = x
            filt_state = self.new_filt_state(obs, filt_state, params)
            return (filt_state, params), filt_state

        filt_state_seq = lax.scan(_step, init=(init_filt_state, formatted_params), xs=obs_seq[1:])[1]


        return tree_prepend(init_filt_state, filt_state_seq)

    def compute_kernel_state_seq(self, filt_seq, formatted_params):
        
        return vmap(self.new_kernel_state, in_axes=(0,None))(tree_droplast(filt_seq), formatted_params)

    def smooth_seq(self, obs_seq, params):
        
        formatted_params = self.format_params(params)

        filt_state_seq = self.compute_filt_state_seq(obs_seq, formatted_params)
        backwd_state_seq = self.compute_kernel_state_seq(filt_state_seq, formatted_params)

        return self.compute_marginals(tree_get_idx(-1, filt_state_seq), backwd_state_seq)

class LinearBackwardSmoother(Smoother):


    def __init__(self, state_dim, filt_dist=Gaussian):

        backwd_kernel_def = ({'homogeneous':False, 'map':'linear'}, (None, True))

        super().__init__(filt_dist, Kernel(state_dim, state_dim, backwd_kernel_def))

    def new_kernel_state(self, filt_state, params):

        A, a, Q = params.transition.map.w, params.transition.map.b, params.transition.scale.cov
        mu, Sigma = filt_state.mean, filt_state.scale.cov
        I = jnp.eye(self.state_dim)

        K = Sigma @ A.T @ inv(A @ Sigma @ A.T + Q)
        C = I - K @ A

        A_back = K 
        a_back = C @ mu - K @ a
        cov_back = C @ Sigma

        return KernelParams(LinearMapParams(A_back, a_back), Scale(cov=cov_back))

    def compute_marginals(self, last_filt_state, backwd_state_seq):


        last_filt_state_mean, last_filt_state_cov = last_filt_state.mean, last_filt_state.scale.cov

        @jit
        def _step(filt_state, backwd_state):
            A_back, a_back, cov_back = backwd_state.map.w, backwd_state.map.b, backwd_state.scale.cov
            smoothed_mean, smoothed_cov = filt_state
            mean = A_back @ smoothed_mean + a_back
            cov = A_back @ smoothed_cov @ A_back.T + cov_back
            return (mean, cov), GaussianParams(mean=mean, scale=Scale(cov=cov))

        marginals = lax.scan(_step, 
                                init=(last_filt_state_mean, last_filt_state_cov), 
                                xs=backwd_state_seq, 
                                reverse=True)[1]
        
        marginals = tree_append(marginals, GaussianParams(mean=last_filt_state_mean, scale=Scale(cov=last_filt_state_cov)))

        return marginals

class LinearGaussianHMM(HMM, LinearBackwardSmoother):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning):

        transition_kernel_def = ({'homogeneous':True, 'map':'linear'}, (transition_matrix_conditionning, False))
        emission_kernel_def =  ({'homogeneous':True, 'map':'linear'}, (None,False))

        HMM.__init__(self, 
                    state_dim, 
                    obs_dim, 
                    transition_kernel_type = lambda state_dim:Kernel(state_dim, state_dim, transition_kernel_def), 
                    emission_kernel_type = lambda state_dim, obs_dim:Kernel(state_dim, obs_dim, emission_kernel_def))

        LinearBackwardSmoother.__init__(self, state_dim)

    def init_filt_state(self, obs, params):

        mean, cov =  Kalman.init(obs, params.prior, params.emission)

        return GaussianParams(mean=mean, scale=Scale(cov=cov))

    def new_filt_state(self, obs, filt_state, params):

        pred_mean, pred_cov = Kalman.predict(filt_state.mean, filt_state.scale.cov, params.transition)
        mean, cov = Kalman.update(pred_mean, pred_cov, obs, params.emission)

        return GaussianParams(mean=mean, scale=Scale(cov=cov))

    def likelihood_seq(self, obs_seq, params):

        return Kalman.filter_seq(obs_seq, self.format_params(params))[-1]
    
    def gaussianize_filt_state(self, filt_state, params):
        return filt_state
    
    def fit_kalman_rmle(self, key, data, optimizer, learning_rate, batch_size, num_epochs):
                
        loss = lambda seq, params: -self.likelihood_seq(seq, params)
        
        key_init_params, key_batcher = random.split(key, 2)
        optimizer = getattr(optax, optimizer)(learning_rate)
        params = self.get_random_params(key_init_params)
        opt_state = optimizer.init(params)
        num_seqs = data.shape[0]

        @jit
        def batch_step(carry, x):
            
            def step(params, opt_state, batch):
                neg_logl_value, grads = vmap(value_and_grad(loss, argnums=1), in_axes=(0,None))(batch, params)
                avg_grads = tree_util.tree_map(jnp.mean, grads)
                updates, opt_state = optimizer.update(avg_grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, jnp.mean(-neg_logl_value)

            data, params, opt_state = carry
            batch_start = x
            batch_obs_seq = lax.dynamic_slice_in_dim(data, batch_start, batch_size)
            params, opt_state, avg_logl_batch = step(params, opt_state, batch_obs_seq)
            return (data, params, opt_state), avg_logl_batch
        
        batch_start_indices = jnp.arange(0, num_seqs, batch_size)

        avg_logls = []

        for _ in range(num_epochs):

            key_batcher, subkey_batcher = random.split(key_batcher, 2)
            
            data = random.permutation(subkey_batcher, data)

            (_, params, opt_state), avg_logl_batches = lax.scan(batch_step, 
                                                                init=(data, params, opt_state), 
                                                                xs=batch_start_indices)
            avg_logls.append(jnp.mean(avg_logl_batches))

        
        return params, avg_logls




class NonLinearGaussianHMM(HMM):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning,
                layers,
                slope,
                num_particles=1000):

        nonlinear_map_forward = partial(neural_map, layers=layers, slope=slope)
        transition_kernel_def = ({'homogeneous':True, 'map':'linear'}, (transition_matrix_conditionning, False))
        emission_kernel_def = ({'homogeneous':True, 'map':'nonlinear'}, nonlinear_map_forward)
        
        HMM.__init__(self, 
                    state_dim, 
                    obs_dim, 
                    transition_kernel_type = lambda state_dim: Kernel(state_dim, state_dim, transition_kernel_def), 
                    emission_kernel_type  = lambda state_dim, obs_dim:Kernel(state_dim, obs_dim, emission_kernel_def))

        self.smc = SMC(self.transition_kernel, self.emission_kernel, self.prior_dist, num_particles)

    def likelihood_seq(self, key, obs_seq, params):

        return self.smc.compute_filt_state_seq(key, 
                            obs_seq, 
                            self.format_params(params))[-1]
    
    def compute_filt_state_seq(self, key, obs_seq, formatted_params):

        return self.smc.compute_filt_state_seq(key, 
                                obs_seq, 
                                formatted_params)[:-1]
        
    def compute_marginals(self, key, filt_seq, formatted_params):

        return self.smc.smooth_from_filt_seq(key, filt_seq, formatted_params)
    
    def smooth_seq(self, key, obs_seq, params):

        key, subkey = random.split(key, 2)

        formatted_params = self.format_params(params)

        filt_seq = self.smc.compute_filt_state_seq(key, 
                                obs_seq, 
                                formatted_params)

        paths = self.smc.smooth_from_filt_seq(subkey, filt_seq, formatted_params)

        return jnp.mean(paths, axis=0), jnp.var(paths, axis=0)
    
    def fit_ffbsi_em(self, key, data, optimizer, learning_rate, batch_size, num_epochs):

        smc = SMC(self.transition_kernel, self.emission_kernel, self.prior_dist, num_particles=100)
        key_init_params, key_batcher = random.split(key, 2)
        optimizer = getattr(optax, optimizer)(learning_rate)
        params = self.get_random_params(key_init_params)
        opt_state = optimizer.init(params)
        num_seqs = data.shape[0]        
        key_batcher, key_montecarlo = random.split(key, 2)
        mc_keys = random.split(key_montecarlo, num_seqs * num_epochs).reshape(num_epochs, num_seqs,-1)

        def e_from_smoothed_paths(theta, smoothed_paths, obs_seq):
            theta = self.format_params(theta)
            def _single_path_e_func(smoothed_path, obs_seq):
                init_val = self.prior_dist.logpdf(smoothed_path[0], theta.prior) \
                    + self.emission_kernel.logpdf(obs_seq[0], smoothed_path[0], theta.emission)
                def _step(prev_particle, particle, obs):
                    return self.transition_kernel.logpdf(particle, prev_particle, theta.transition) + \
                        self.emission_kernel.logpdf(obs, particle, theta.emission)
                return init_val + jnp.sum(vmap(_step)(smoothed_path[:-1], smoothed_path[1:], obs_seq[1:]))
            return jnp.mean(vmap(_single_path_e_func, in_axes=(0,None))(smoothed_paths, obs_seq))
        
        @jit
        def batch_step(carry, x):
            
            def step(prev_theta, opt_state, batch, keys):

                def e_step(key, obs_seq, theta):
                    
                    formatted_prev_theta = self.format_params(prev_theta)
                    
                    key_fwd, key_backwd = random.split(key, 2)
                    
                    filt_seq, logl_value = smc.compute_filt_state_seq(key_fwd, 
                            obs_seq, 
                            formatted_prev_theta)

                    smoothed_paths = smc.smooth_from_filt_seq(key_backwd, filt_seq, formatted_prev_theta)

                    return -e_from_smoothed_paths(theta, smoothed_paths, obs_seq), logl_value

                grads, logl_values = vmap(grad(e_step, argnums=2, has_aux=True), in_axes=(0,0, None))(keys, batch, prev_theta)
                avg_grads = tree_util.tree_map(jnp.mean, grads)
                updates, opt_state = optimizer.update(avg_grads, opt_state, prev_theta)
                theta = optax.apply_updates(prev_theta, updates)
                return theta, opt_state, jnp.mean(logl_values)

            data, params, opt_state, subkeys_epoch = carry
            batch_start = x
            batch_obs_seq = lax.dynamic_slice_in_dim(data, batch_start, batch_size)
            batch_keys = lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, batch_size)
            params, opt_state, avg_logl_batch = step(params, opt_state, batch_obs_seq, batch_keys)
            return (data, params, opt_state, subkeys_epoch), avg_logl_batch
        
        batch_start_indices = jnp.arange(0, num_seqs, batch_size)

        avg_logls = []

        for epoch_nb in tqdm(range(num_epochs)):
            mc_keys_epoch = mc_keys[epoch_nb]
            key_batcher, subkey_batcher = random.split(key_batcher, 2)
            
            data = random.permutation(subkey_batcher, data)

            (_, params, opt_state, _), avg_logl_batches = lax.scan(batch_step, 
                                                                init=(data, params, opt_state, mc_keys_epoch), 
                                                                xs=batch_start_indices)
            avg_logls.extend([logl_batch for logl_batch in avg_logl_batches])

        
        return params, avg_logls






class NeuralLinearBackwardSmoother(LinearBackwardSmoother):


    @staticmethod
    def johnson_update_forward(obs, pred_state:GaussianParams, layers, out_dim):

        rec_net = hk.nets.MLP((*layers, out_dim),
                    w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                    b_init=hk.initializers.RandomNormal(),
                    activation=xtanh(0.1),
                    activate_final=False)

        # R_prec_diagonal = hk.get_parameter('R_prec', 
        #                 shape=(pred_state.mean.shape[0],), 
        #                 dtype=jnp.float64, init=hk.initializers.Constant(1 / HMM.default_emission_base_scale**2))

        # R_prec = jnp.diag(R_prec_diagonal)


        out = rec_net(obs)
        eta1, log_prec_diag = jnp.split(out,2)
        eta2 = - 0.5 * jnp.diag(nn.softplus(log_prec_diag))
        return GaussianParams(eta1 = eta1 + pred_state.eta1, eta2 = eta2 + pred_state.eta2)

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_kernel_matrix_conditionning='diagonal', 
                update_layers=(100,), 
                use_johnson=False,
                prior_dist=Gaussian, 
                filt_dist=Gaussian):
        
        super().__init__(state_dim, filt_dist)

        self.state_dim, self.obs_dim = state_dim, obs_dim 

        self.prior_dist:Gaussian = prior_dist
        transition_kernel_def = ({'homogeneous':True, 'map':'linear'}, (transition_kernel_matrix_conditionning, True))
        self.transition_kernel = Kernel(state_dim, state_dim, transition_kernel_def)

        d = state_dim
        self.filt_state_shape = d + d*(d+1) // 2

        if use_johnson: 
            self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(self.johnson_update_forward, 
                                                                                layers=update_layers, 
                                                                                out_dim=self.filt_state_shape)))
        else: 
            self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(self.filt_update_forward, 
                                                                    layers=update_layers, 
                                                                    out_dim=self.filt_state_shape)))
                                
    def get_random_params(self, key):

        subkeys = random.split(key, 3)

        dummy_obs = jnp.empty((self.obs_dim,))

        prior_params = self.prior_dist.get_random_params(subkeys[0], self.state_dim, default_mean=0, default_base_scale=HMM.default_prior_base_scale)
        transition_params = self.transition_kernel.get_random_params(subkeys[1], default_base_scale=HMM.default_transition_base_scale)
        dummy_pred_state = GaussianParams(mean=jnp.ones((self.state_dim,)), scale=Scale(chol=jnp.eye(self.state_dim)))
        
        filt_update_params = self.filt_update_init_params(subkeys[2], dummy_obs, dummy_pred_state)

        return NeuralLinearBackwardSmootherParams(prior_params, transition_params, filt_update_params)

    def init_filt_state(self, obs, params):

        return self.filt_update_apply(params.filt_update, obs, params.prior)

    def new_filt_state(self, obs, filt_state, params):

        pred_mean, pred_cov = Kalman.predict(filt_state.mean, filt_state.scale.cov, params.transition)

        pred_state = GaussianParams(mean=pred_mean, scale=Scale(cov=pred_cov))

        return self.filt_update_apply(params.filt_update, obs, pred_state)
    
    def format_params(self, params):
        return NeuralLinearBackwardSmootherParams(self.prior_dist.format_params(params.prior),
                                                self.transition_kernel.format_params(params.transition),
                                                params.filt_update)

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', sum(len(jnp.atleast_1d(leaf)) for leaf in tree_leaves(params)))
        print('-- in prior + predict + backward:', sum(len(jnp.atleast_1d(leaf)) for leaf in tree_leaves((params.prior, params.transition))))
        print('-- in update:', sum(len(jnp.atleast_1d(leaf)) for leaf in tree_leaves(params.filt_update)))

# class NeuralBackwardSmoother(Smoother):

        
#     def __init__(self, state_dim, obs_dim,
#                 transition_kernel_matrix_conditionning='diagonal',
#                 update_layers=(10,),
#                 backwd_layers=(10,),
#                 prior_dist=Gaussian, 
#                 filt_dist=Gaussian,
#                 backwd_dist=Gaussian):


#         self.state_dim, self.obs_dim = state_dim, obs_dim
#         self.prior_dist = prior_dist 

#         d = state_dim
#         self.filt_state_shape = d + d*(d+1) // 2
#         transition_kernel_def = ({'homogeneous':True, 'map':'linear'}, (transition_kernel_matrix_conditionning, True))
#         self.transition_kernel = Kernel(state_dim, state_dim, transition_kernel_def)

#         backwd_kernel_def = ({'homogeneous':False, 'map':'nonlinear'}, 
#                         (partial(Smoother.backwd_kernel_map_forward, layers=backwd_layers), self.filt_state_shape))

#         super().__init__(filt_dist, Kernel(state_dim, state_dim, backwd_kernel_def, backwd_dist))

#         self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(Smoother.filt_update_forward, 
#                                                                                 layers=update_layers, 
#                                                                                 out_dim=self.filt_state_shape)))

#     def get_random_params(self, key):


#         subkeys = random.split(key, 4)

#         dummy_obs = jnp.empty((self.obs_dim,))

#         prior_params = self.prior_dist.get_random_params(subkeys[0], self.state_dim, HMM.default_prior_base_scale)
#         transition_params = self.transition_kernel.get_random_params(subkeys[1], HMM.default_transition_base_scale)
#         filt_update_params = self.filt_update_init_params(subkeys[2], dummy_obs, jnp.empty((self.filt_state_shape,)))        
#         backwd_map_params = self.kernel.get_random_params(subkeys[3])

#         return NeuralBackwardSmootherParams(prior_params, transition_params, filt_update_params, backwd_map_params)

#     def format_params(self, params):
#         formatted_prior_params = self.prior_dist.format_params(params.prior)
#         formatted_transition_params = self.transition_kernel.format_params(params.transition)

#         return NeuralBackwardSmootherParams(formatted_prior_params, formatted_transition_params, params.filt_update, params.backwd_map)


#     def init_filt_state(self, obs, params):

#         return self.filt_update_apply(params.filt_update, obs, params.prior)

#     def new_filt_state(self, obs, filt_state, params):

#         pred_mean, pred_cov = Kalman.predict(filt_state.mean, filt_state.scale.cov, params.transition)
#         pred_state = vec_from_gaussian_params(GaussianParams(mean=pred_mean, scale=Scale(cov=pred_cov)), self.state_dim)

#         filt_state  = self.filt_update_apply(params.filt_update, obs, pred_state)

#         return gaussian_params_from_vec(filt_state, self.state_dim)
    
#     def new_kernel_state(self, filt_state, params):
#         return BackwardState(vec_from_gaussian_params(filt_state, self.state_dim), params.backwd_map)

#     def compute_marginals(self, last_filt_state, backwd_state_seq):
#         pass 

#     def print_num_params(self):
#         params = self.get_random_params(random.PRNGKey(0))
#         print('Num params:', sum(len(jnp.atleast_1d(leaf)) for leaf in tree_leaves(params)))
#         print('-- in prior + predict/transition', sum(len(jnp.atleast_1d(leaf)) for leaf in tree_leaves((params.prior, params.transition))))
#         print('-- in backward map',  sum(len(jnp.atleast_1d(leaf)) for leaf in tree_leaves((params.backwd_map))))
#         print('-- in update:', sum(len(jnp.atleast_1d(leaf)) for leaf in tree_leaves(params.filt_update)))


# class NeuralLinearForwardSmoother(Smoother):

#     def __init__(self, state_dim, obs_dim,            
#                 transition_kernel_matrix_conditionning='diagonal', 
#                 update_layers = (100,), 
#                 prior_dist=Gaussian, 
#                 filt_dist=Gaussian):

#         fwd_kernel_def = ({'homogeneous':False, 'map':'linear'}, None)

#         super().__init__(filt_dist, Kernel(state_dim, state_dim, fwd_kernel_def))


#         self.prior_dist:Gaussian = prior_dist
#         transition_kernel_def = ({'homogeneous':True, 'map':'linear'}, transition_kernel_matrix_conditionning)
#         self.transition_kernel = Kernel(state_dim, state_dim, transition_kernel_def)
#         self.filt_dist:Gaussian = filt_dist

#         d = state_dim
#         self.filt_state_shape = d + d*(d+1) // 2
        
#         self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(filt_update_forward, 
#                                                                                 layers=update_layers, 
#                                                                                 out_dim=self.filt_state_shape))) 
#     def init_filt_state(self, obs, params):

#         pred_state = vec_from_gaussian_params(params.prior, self.state_dim)

#         filt_state = self.filt_update_apply(params.filt_update, obs, pred_state)
#         return gaussian_params_from_vec(filt_state, self.state_dim)

#     def new_filt_state(self, obs, filt_state, params):

#         pred_mean, pred_cov = Kalman.predict(filt_state.mean, filt_state.scale.cov, params.transition)
#         pred_state = vec_from_gaussian_params(GaussianParams(pred_mean, Scale(cov=pred_cov)), self.state_dim)

#         filt_state  = self.filt_update_apply(params.filt_update, obs, pred_state)

#         return gaussian_params_from_vec(filt_state, self.state_dim)

#     def new_kernel_state(self, filt_state, params):
#         raise NotImplementedError

#     def format_params(self, params):
#         return NeuralLinearForwardSmootherParams(self.prior_dist.format_params(params.prior),
#                                                 self.transition_kernel.format_params(params.transition),
#                                                 params.filt_update)





        

            
        



        

