from abc import ABCMeta, abstractmethod
from random import gauss
from jax import numpy as jnp, random, value_and_grad, tree_util, grad, config
from jax.tree_util import tree_leaves
from backward_ica.kalman import Kalman
from backward_ica.smc import SMC
import haiku as hk
from jax import lax, vmap
from .utils import *
from jax.scipy.stats.multivariate_normal import logpdf as gaussian_logpdf, pdf as gaussian_pdf
from jax.scipy.stats.t import logpdf as student_logpdf, pdf as student_pdf

from functools import partial
from jax import nn
import optax
config.update('jax_enable_x64',True)
import copy 
_conditionnings = {'diagonal':lambda param, d: jnp.diag(param),
                'sym_def_pos': lambda param, d: mat_from_chol_vec(param, d) + jnp.eye(d),
                None:lambda x, d:x,
                'init_invertible': lambda x,d:x + jnp.eye(d)}

def xtanh(slope):
    return lambda x: jnp.tanh(x) + slope*x

# def l2normalize(W, axis=0):
#     """Normalizes MLP weight matrices.
#     Args:
#         W (matrix): weight matrix.
#         axis (int): axis over which to normalize.
#     Returns:
#         Matrix l2 normalized over desired axis.
#     """
#     l2norm = jnp.sqrt(jnp.sum(W*W, axis, keepdims=True))
#     W = W / l2norm
#     return W

# def unif_nica_layer(N, M, key, iter_4_cond=1e4):

#     def _gen_matrix(N, M, key):
#         A = random.uniform(key, (N, M), minval=0., maxval=2.) - 1.
#         A = l2normalize(A)
#         _cond = jnp.linalg.cond(A)
#         return A, _cond

#     # generate multiple matrices
#     keys = random.split(key, iter_4_cond)
#     A, conds = vmap(_gen_matrix, (None, None, 0))(N, M, keys)
#     target_cond = jnp.percentile(conds, 25)
#     target_idx = jnp.argmin(jnp.abs(conds-target_cond))
#     return A[target_idx]

# def init_nica_params(key, N, obs_dim, nonlin_layers):
#     '''BEWARE: Assumes factorized distribution
#         and equal width in all hidden layers'''

#     layer_sizes = [N] + [obs_dim]*nonlin_layers + [obs_dim]
#     keys = random.split(key, len(layer_sizes)-1)
#     return [unif_nica_layer(n, m, k) for (n, m, k)
#             in zip(layer_sizes[:-1], layer_sizes[1:], keys))

# def nica_mlp(params, s, slope=0.1):
#     """Forward pass for encoder MLP for estimating nonlinear mixing function.
#     Args: (OLD; IGNORE)
#         params (list): nested list where each element is a list of weight
#             matrix and bias for a given layer. e.g. [[W_0, b_0], [W_1, b_1]].
#         inputs (matrix): input data.
#         slope (float): slope to control the nonlinearity of the activation
#             function.
#     Returns:
#         Outputs f(s)
#     """
#     act = xtanh(slope)
#     params = list(params.values())

#     z = s
#     if len(params) > 1:
#         hidden_params = params[:-1]
#         for i in range(len(hidden_params)):
#             z = act(z@hidden_params[i])
#     A_final = params[-1]
#     z = z@A_final
#     return z

def neural_map(input, layers, slope, out_dim):

    net = hk.nets.MLP((*layers, out_dim), 
                    activate_final=True, 
                    w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                    b_init=hk.initializers.RandomNormal(),
                    activation=nn.relu)

    return net(input)

def neural_map_noninjective(input, layers, slope, out_dim):

    net = hk.nets.MLP((*layers, out_dim), 
                    with_bias=False, 
                    activate_final=True, 
                    activation=nn.tanh)
    x = net(input)
    return jnp.cos(x)

def chaotic_map(x, grid_size, gamma, tau, out_dim):
    linear_map = hk.Linear(out_dim, 
                        with_bias=False, 
                        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'normal'))
    return x + grid_size * (-x + gamma * linear_map(nn.tanh(x))) / tau

def linear_map_apply(map_params, input):
    out =  jnp.dot(map_params.w, input)
    return out + jnp.broadcast_to(map_params.b, out.shape)

def linear_map_init_params(key, dummy_in, out_dim, conditionning, bias, range_params):

    key_w, key_b = random.split(key, 2)

    if conditionning == 'diagonal':
        w = random.uniform(key_w, (out_dim,), minval=range_params[0], maxval=range_params[1])
    elif conditionning == 'sym_def_pos':
        d = out_dim 
        w = random.uniform(key_w, ((d*(d+1)) // 2,), minval=range_params[0], maxval=range_params[1])
    elif conditionning == 'init_invertible':
        w = random.uniform(key_w, (out_dim, len(dummy_in)), minval=range_params[0], maxval=range_params[1])
        w = w @ w.T
    else: 
        w = random.uniform(key_w, (out_dim, len(dummy_in)), minval=range_params[0], maxval=range_params[1])
        
        
    if bias: 
        b = random.uniform(key_b, (out_dim,))
        return LinearMapParams(w=w, b=b)
    else: 
        return LinearMapParams(w=w)

def linear_map_format_params(params, conditionning_func, d):

    w = conditionning_func(params.w, d)
    
    if not hasattr(params, 'b'):
        b = jnp.zeros((d,))
    else: 
        b = params.b

    return LinearMapParams(w,b)


class Gaussian: 

    @staticmethod
    def sample(key, params):
        return params.mean + params.scale.cov_chol @ random.normal(key, (params.mean.shape[0],))
    
    @staticmethod
    def logpdf(x, params):
        return gaussian_logpdf(x, params.mean, params.scale.cov)
    
    @staticmethod
    def pdf(x, params):
        return gaussian_pdf(x, params.mean, params.scale.cov)

    @staticmethod
    def get_random_params(key, dim):
        
        subkeys = random.split(key,2)

        mean = random.uniform(subkeys[0], shape=(dim,), minval=-1, maxval=1)
        return GaussianParams(mean, Scale.get_random(key, dim, HMM.parametrization))

    @staticmethod
    def format_params(params):
        return GaussianParams(mean=params.mean, scale=Scale.format(params.scale))

    @staticmethod
    def get_random_noise_params(key, dim):
        return GaussianNoiseParams(Scale.get_random(key, dim, HMM.parametrization))

    @staticmethod
    def format_noise_params(noise_params):
        return GaussianNoiseParams(Scale.format(noise_params.scale))

    @staticmethod
    def KL(params_0, params_1):
        mu_0, sigma_0 = params_0.mean, params_0.scale.cov
        mu_1, sigma_1, inv_sigma_1 = params_1.mean, params_1.scale.cov, params_1.scale.prec 
        d = mu_0.shape[0]

        return 0.5 * (jnp.trace(inv_sigma_1 @ sigma_0) \
                    + (mu_1 - mu_0).T @ inv_sigma_1 @ (mu_1 - mu_0) 
                    - d \
                    + jnp.log(jnp.linalg.det(sigma_1) / jnp.linalg.det(sigma_0)))

    @staticmethod
    def squared_wasserstein_2(params_0, params_1):
        mu_0, sigma_0 = params_0.mean, params_0.scale.cov
        mu_1, sigma_1 = params_1.mean, params_1.scale.cov
        sigma_0_half = jnp.sqrt(sigma_0)
        return jnp.linalg.norm(mu_0 - mu_1, ord=2) ** 2 \
                + jnp.trace(sigma_0 + sigma_1  - 2*jnp.sqrt(sigma_0_half @ sigma_1 @ sigma_0_half))

class Student: 

    # @staticmethod
    # def sample(key, params):
    #     key, subkey = random.split(key, 2)
    #     y = params.scale.cov_chol @ random.normal(key, shape=(params.mean.shape[0],))
    #     u = random.gamma(subkey, a=params.df / 2) / 0.5 

    #     return params.mean + jnp.sqrt(params.df / u) * y
    
    def sample(key, params):
        return params.mean + params.scale.cov_chol @ random.t(key, params.df, shape=(params.mean.shape[0],))

    @staticmethod
    def logpdf(x, params):

        return vmap(student_logpdf, in_axes=(0, None, 0, 0))(x, params.df, params.mean, jnp.diag(params.scale.cov_chol)).sum()

    
    @staticmethod
    def pdf(x, params):
        return vmap(student_pdf, in_axes=(0, None, 0, 0))(x, params.df, params.mean, jnp.diag(params.scale.cov_chol)).prod()


    @staticmethod
    def get_random_params(key, dim):
        
        subkeys = random.split(key,3)


        mean = random.uniform(subkeys[0], shape=(dim,), minval=-1, maxval=1)
        df = random.randint(subkeys[1], shape=(1,), minval=1, maxval=10)
        scale = Scale.get_random(subkeys[3], dim, HMM.parametrization)
        return StudentParams(mean=mean, 
                            df=df, 
                            scale=scale)

    @staticmethod
    def format_params(params):
        return StudentParams(mean=params.mean, df=params.df, scale=Scale.format(params.scale))

    @staticmethod
    def get_random_noise_params(key, dim):
        subkeys = random.split(key, 2)
        df = random.randint(subkeys[1], shape=(1,), minval=1, maxval=10)
        scale = Scale.get_random(subkeys[1], dim, HMM.parametrization)
        return StudentNoiseParams(df, scale)

    @staticmethod 
    def format_noise_params(noise_params):
        return StudentNoiseParams(noise_params.df, Scale.format(noise_params.scale))

class Kernel:

    @staticmethod
    def linear_gaussian(matrix_conditonning, bias, range_params):
        transition_kernel_def = {'map_type':'linear',
                        'map_info' : {'conditionning': matrix_conditonning, 
                                    'bias': bias,
                                    'range_params':range_params}}
        return lambda state_dim, obs_dim: Kernel(state_dim, obs_dim, transition_kernel_def)
                                                                 
    def __init__(self,
                in_dim, 
                out_dim,
                map_def, 
                noise_dist=Gaussian):

        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.noise_dist = noise_dist

        self.map_type = map_def['map_type']



        if noise_dist == Gaussian:
            self.format_output = lambda mean, noise, params: GaussianParams(mean, noise.scale)
            self.params_type = GaussianNoiseParams
            
        elif noise_dist == Student:
            self.format_output = lambda mean, noise, params: StudentParams(mean=mean, df=noise.df, scale=noise.scale)
            self.params_type = StudentNoiseParams

        if self.map_type == 'linear':

            apply_map = lambda params, input: (linear_map_apply(params.map, input), params.noise)

            init_map_params = partial(linear_map_init_params, out_dim=out_dim, 
                                    conditionning=map_def['map_info']['conditionning'], 
                                    bias=map_def['map_info']['bias'], 
                                    range_params=map_def['map_info']['range_params'])

            get_random_map_params = lambda key: init_map_params(key, jnp.empty((self.in_dim,)))

            format_map_params = partial(linear_map_format_params, 
                                        conditionning_func=_conditionnings[map_def['map_info']['conditionning']],
                                        d=self.out_dim)


        elif self.map_type == 'nonlinear':
            if map_def['map_info']['homogeneous']: 
        
                init_map_params, nonlinear_apply_map = hk.without_apply_rng(hk.transform(partial(map_def['map'], 
                                                                                    out_dim=out_dim)))                                 
                apply_map = lambda params, input: (nonlinear_apply_map(params.map, input), params.noise)

                get_random_map_params = lambda key: init_map_params(key, jnp.empty((self.in_dim,)))

                format_map_params = lambda x:x
                
            else: 
                
                init_map_params, nonlinear_apply_map = hk.without_apply_rng(hk.transform(partial(map_def['map'], 
                                                                                state_dim=out_dim)))
                
                def apply_map(params, input):
                    mean, scale = nonlinear_apply_map(params.inner.map, params.varying, input)
                    return (mean, GaussianNoiseParams(scale))

                get_random_map_params = lambda key: init_map_params(key, 
                                                                    jnp.empty((map_def['map_info']['varying_params_shape'],)), 
                                                                    jnp.empty((self.in_dim,)))
                
                format_map_params = lambda x:x
        


        self._apply_map = apply_map 
        self._get_random_map_params = get_random_map_params
        self._format_map_params = format_map_params 
        self._get_random_noise_params = lambda key: noise_dist.get_random_noise_params(key, self.out_dim)

    def map(self, state, params):
        mean, scale = self._apply_map(params, state)
        return self.format_output(mean, scale, params)
    
    def sample(self, key, state, params):
        return self.noise_dist.sample(key, self.map(state, params))

    def logpdf(self, x, state, params):
        return self.noise_dist.logpdf(x, self.map(state, params))
    
    def pdf(self, x, state, params):
        return self.noise_dist.pdf(x, self.map(state, params))

    def get_random_params(self, key):
        key, subkey = random.split(key, 2)
        return KernelParams(self._get_random_map_params(key), self._get_random_noise_params(subkey))

    def format_params(self, params):
        return KernelParams(self._format_map_params(params.map), 
                            self.noise_dist.format_noise_params(params.noise))


class HMM: 

    parametrization = 'cov_chol'

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
        
    def sample_multiple_sequences(self, key, params, num_seqs, seq_length, single_split_seq=False, loaded_data=None):

        if loaded_data is not None: 
            state_seq, obs_seq = jnp.load(loaded_data[0]).astype(jnp.float64), jnp.load(loaded_data[1]).astype(jnp.float64)
            print('Sequences loaded.')
            if single_split_seq: 
                return jnp.array(jnp.split(state_seq, num_seqs)), jnp.array(jnp.split(obs_seq, num_seqs))
            else: 
                return state_seq[:seq_length][jnp.newaxis,:], obs_seq[:seq_length][jnp.newaxis,:]
        else: 
            if single_split_seq: 
                state_seq, obs_seq = self.sample_seq(key, params, num_seqs*seq_length)
                return jnp.array(jnp.split(state_seq, num_seqs)), jnp.array(jnp.split(obs_seq, num_seqs))
            else: 
                key, *subkeys = random.split(key, num_seqs+1)
                sampler = vmap(self.sample_seq, in_axes=(0, None, None))
                return sampler(jnp.array(subkeys), params, seq_length)

    def get_random_params(self, key, params_to_set=None):
        key_prior, key_transition, key_emission = random.split(key, 3)

        prior_params = self.prior_dist.get_random_params(key_prior, 
                                                        self.state_dim)

        transition_params = self.transition_kernel.get_random_params(key_transition)
        emission_params = self.emission_kernel.get_random_params(key_emission)
        params = HMMParams(prior_params, 
                        transition_params, 
                        emission_params)
        if params_to_set is not None: 
            params = self.set_params(params, params_to_set)
        return params 
        
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
        print('Num params:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params)))
        print('-- in prior + predict:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.prior, params.transition))))
        print('-- in update:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params.emission)))

    def set_params(self, params, args):
        new_params = copy.deepcopy(params)
        for k,v in vars(args).items():         
            if k == 'default_prior_mean':
                new_params.prior.mean = v * jnp.ones_like(params.prior.mean)
            elif k == 'default_prior_base_scale':
                new_params.prior.scale = Scale.set_default(params.prior.scale, v, HMM.parametrization)
            elif k == 'default_transition_base_scale': 
                new_params.transition.noise.scale = Scale.set_default(params.transition.noise.scale, v, HMM.parametrization)
            elif k == 'default_emission_base_scale': 
                new_params.emission.noise.scale = Scale.set_default(params.emission.noise.scale, v, HMM.parametrization)
            elif k == 'default_emission_df':
                new_params.emission.noise.df = v
            elif k == 'default_emission_matrix' and hasattr(new_params.emission.map, 'w'):
                new_params.emission.map.w = v * jnp.ones_like(params.emission.map.w)
            elif (k == 'default_transition_matrix') and (self.transition_kernel.map_type != 'linear'):
                if (type(v) == str): new_params.transition.map['linear']['w'] = jnp.load(v).astype(jnp.float64)
        return new_params

class BackwardSmoother(metaclass=ABCMeta):

    @staticmethod
    def filt_update_forward(obs, prev_state, layers, state_dim):

        d = state_dim 

        out_dim = d + (d*(d+1)) // 2
        gru = hk.DeepRNN([hk.GRU(hidden_size) for hidden_size in (*layers,)])
        projection = hk.Linear(out_dim, 
                    w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                    b_init=hk.initializers.RandomNormal(),)
        out, new_state = gru(obs, prev_state)
        out = projection(out)

        return GaussianParams.from_vec(out, d, diag=False, chol_add=empty_add), new_state


    @staticmethod
    def backwd_update_forward(varying_params, next_state, layers, state_dim):

        d = state_dim
        out_dim = d + (d * (d+1)) // 2

        net = hk.nets.MLP((*layers, out_dim),
                    w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                    b_init=hk.initializers.RandomNormal(),
                    activation=nn.tanh,
                    activate_final=False)
        
        out = net(jnp.concatenate((varying_params, next_state)))

        out = GaussianParams.from_vec(out, d, diag=False)

        return out.mean, out.scale

    @staticmethod
    def johnson_update_forward(obs, pred_state:GaussianParams, layers, state_dim):

        d = state_dim 
        rec_net = hk.nets.MLP((*layers, 2*d),
                    w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                    b_init=hk.initializers.RandomNormal(),
                    activation=nn.tanh,
                    activate_final=False)


        out = rec_net(obs)
        eta1, log_prec_diag = jnp.split(out,2)
        eta2 = - 0.5 * jnp.diag(nn.softplus(log_prec_diag))
        filt_state = GaussianParams(eta1 = eta1 + pred_state.eta1, 
                                    eta2 = eta2 + pred_state.eta2)

        return FiltState(filt_state, filt_state)


    def __init__(self, filt_dist, kernel):

        self.filt_dist:Gaussian = filt_dist
        self.backwd_kernel:Kernel = kernel

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
    def compute_marginals(self, *args):

        raise NotImplementedError

    @abstractmethod
    def smooth_seq(self, *args):
        raise NotImplementedError

    def compute_filt_state_seq(self, obs_seq, formatted_params):
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
        
        return vmap(self.new_backwd_state, in_axes=(0,None))(tree_droplast(filt_seq), formatted_params)

class LinearBackwardSmoother(BackwardSmoother):


    def __init__(self, state_dim, filt_dist=Gaussian):

        backwd_kernel_def = {'map_type':'linear',
                            'map_info' : {'conditionning': None, 
                                        'bias': True,
                                        'range_params':(0,1)}}

        super().__init__(filt_dist, Kernel(state_dim, state_dim, backwd_kernel_def))

    def new_backwd_state(self, filt_state, params):

        A, a, Q = params.transition.map.w, params.transition.map.b, params.transition.noise.scale.cov
        mu, Sigma = filt_state.out.mean, filt_state.out.scale.cov
        I = jnp.eye(self.state_dim)

        K = Sigma @ A.T @ inv(A @ Sigma @ A.T + Q)
        C = I - K @ A

        A_back = K 
        a_back = C @ mu - K @ a
        cov_back = C @ Sigma

        return KernelParams(LinearMapParams(A_back, a_back), GaussianNoiseParams(Scale(cov=cov_back)))

    def compute_marginals(self, last_filt_state, backwd_state_seq):
        last_filt_state_mean, last_filt_state_cov = last_filt_state.out.mean, last_filt_state.out.scale.cov

        @jit
        def _step(filt_state, backwd_state):
            A_back, a_back, cov_back = backwd_state.map.w, backwd_state.map.b, backwd_state.noise.scale.cov
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

    def compute_fixed_lag_marginals(self, filt_state_seq, backwd_state_seq, lag):
        
        def _compute_fixed_lag_marginal(init, x):

            lagged_filt_state, backwd_state_subseq = x

            lagged_filt_state_mean, lagged_filt_state_cov = lagged_filt_state.out.mean, lagged_filt_state.out.scale.cov

            @jit
            def _marginal_step(current_marginal, backwd_state):
                A_back, a_back, cov_back = backwd_state.map.w, backwd_state.map.b, backwd_state.noise.scale.cov
                smoothed_mean, smoothed_cov = current_marginal
                mean = A_back @ smoothed_mean + a_back
                cov = A_back @ smoothed_cov @ A_back.T + cov_back
                return (mean, cov), None

            marginal = lax.scan(_marginal_step, 
                                    init=(lagged_filt_state_mean, lagged_filt_state_cov), 
                                    xs=backwd_state_subseq, 
                                    reverse=True)[0]

            return None, GaussianParams(mean=marginal[0], scale=Scale(cov=marginal[1]))

        return lax.scan(_compute_fixed_lag_marginal, 
                            init=None, 
                            xs=(tree_get_slice(lag, None, filt_state_seq), tree_get_strides(lag, backwd_state_seq)))[1]

    def filt_seq(self, obs_seq, params):
        
        filt_state_seq = self.compute_filt_state_seq(obs_seq, self.format_params(params)).out
        return vmap(lambda x:x.mean)(filt_state_seq), vmap(lambda x:x.scale.cov)(filt_state_seq)
    
    def smooth_seq(self, obs_seq, params, lag=None):
        
        formatted_params = self.format_params(params)

        filt_state_seq = self.compute_filt_state_seq(obs_seq, formatted_params)
        backwd_state_seq = self.compute_kernel_state_seq(filt_state_seq, formatted_params)

        if lag is None: 
            marginals = self.compute_marginals(tree_get_idx(-1, filt_state_seq), backwd_state_seq)
        else: 
            marginals = self.compute_fixed_lag_marginals(filt_state_seq, backwd_state_seq, lag)

        return marginals.mean, marginals.scale.cov     

    def smooth_seq_at_multiple_timesteps(self, obs_seq, params, slices):
        formatted_params = self.format_params(params)

        filt_state_seq = self.compute_filt_state_seq(obs_seq, formatted_params)
        backwd_state_seq = self.compute_kernel_state_seq(filt_state_seq, formatted_params)


        def smooth_up_to_timestep(timestep):
            marginals = self.compute_marginals(tree_get_idx(timestep, filt_state_seq), tree_get_slice(0, timestep-1, backwd_state_seq))
            return marginals.mean, marginals.scale.cov
        means, covs = [], []

        for timestep in slices:
            mean, cov = smooth_up_to_timestep(timestep)
            means.append(mean)
            covs.append(cov)
            
        return means, covs  

class LinearGaussianHMM(HMM, LinearBackwardSmoother):

    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_matrix_conditionning=None,
                range_transition_map_params=(0,1),
                transition_bias=False,
                emission_bias=False):

        transition_kernel = Kernel.linear_gaussian(transition_matrix_conditionning, 
                                                    transition_bias, 
                                                    range_transition_map_params)
                                        
        emission_kernel = Kernel.linear_gaussian(None, emission_bias, (0,1))                     

        HMM.__init__(self, 
                    state_dim, 
                    obs_dim, 
                    transition_kernel_type = lambda state_dim: transition_kernel(state_dim, state_dim), 
                    emission_kernel_type = emission_kernel)

        LinearBackwardSmoother.__init__(self, state_dim)

    def init_filt_state(self, obs, params):

        mean, cov =  Kalman.init(obs, params.prior, params.emission)

        filt_state = GaussianParams(mean=mean, scale=Scale(cov=cov))
        return FiltState(filt_state, filt_state)

    def new_filt_state(self, obs, filt_state, params):

        pred_mean, pred_cov = Kalman.predict(filt_state.out.mean, filt_state.out.scale.cov, params.transition)
        mean, cov = Kalman.update(pred_mean, pred_cov, obs, params.emission)
        filt_state = GaussianParams(mean=mean, scale=Scale(cov=cov))
        return FiltState(filt_state, filt_state)

    def likelihood_seq(self, obs_seq, params):

        return Kalman.filter_seq(obs_seq, self.format_params(params))[-1]
    
    def fit_kalman_rmle(self, key, data, optimizer, learning_rate, batch_size, num_epochs, theta_star=None):
                
        
        
        key_init_params, key_batcher = random.split(key, 2)
        base_optimizer = getattr(optax, optimizer)(learning_rate)
        optimizer = base_optimizer
        # optimizer = optax.masked(base_optimizer, mask=HMMParams(prior_mask, transition_mask, emission_mask))

        params = self.get_random_params(key_init_params)

        prior_scale = theta_star.prior.scale
        transition_scale = theta_star.transition.noise.scale
        emission_params = theta_star.emission

        def build_params(params):
            return HMMParams(GaussianParams(mean=params[0], scale=prior_scale), 
                            KernelParams(params[1], transition_scale), 
                            emission_params)
        
        params = (params.prior.mean, params.transition.map)

        # build_params = lambda x:x
        loss = lambda seq, params: -self.likelihood_seq(seq, build_params(params))

        # if theta_star is not None: 
        #     params = theta_star
        opt_state = optimizer.init(params)
        num_seqs = data.shape[0]
        
        @jit
        def batch_step(carry, x):
            
            def step(params, opt_state, batch):
                neg_logl_value, grads = vmap(value_and_grad(loss, argnums=1), in_axes=(0,None))(batch, params)
                avg_grads = tree_util.tree_map(jnp.mean, grads)
                updates, opt_state = optimizer.update(avg_grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return params, opt_state, -neg_logl_value.sum()

            data, params, opt_state = carry
            batch_start = x
            batch_obs_seq = lax.dynamic_slice_in_dim(data, batch_start, batch_size)
            params, opt_state, avg_logl_batch = step(params, opt_state, batch_obs_seq)
            return (data, params, opt_state), avg_logl_batch
        
        batch_start_indices = jnp.arange(0, num_seqs, batch_size)

        avg_logls = []
        all_params = []

        for epoch_nb in tqdm(range(num_epochs), 'Epoch'):

            key_batcher, subkey_batcher = random.split(key_batcher, 2)
            
            data = random.permutation(subkey_batcher, data)

            (_, params, opt_state), avg_logl_batches = lax.scan(batch_step, 
                                                                init=(data, params, opt_state), 
                                                                xs=batch_start_indices)
            avg_logls.append(avg_logl_batches.sum())
            all_params.append(params)

        best_optim = jnp.argmax(jnp.array(avg_logls))
        print(f'Best fit is epoch {best_optim} with logl {avg_logls[best_optim]}.')
        best_params = all_params[best_optim]
        
        return build_params(best_params), avg_logls, best_optim
    
    def compute_filt_state_seq(self, obs_seq, formatted_params):
        formatted_params.compute_covs()
        return super().compute_filt_state_seq(obs_seq, formatted_params)

class NonLinearHMM(HMM):

    @staticmethod
    def linear_transition_with_nonlinear_emission(args):
        if args.injective:
            nonlinear_map_forward = partial(neural_map, layers=args.emission_map_layers, slope=args.slope)
        else: 
            nonlinear_map_forward = partial(neural_map_noninjective, layers=args.emission_map_layers, slope=args.slope)
            
        transition_kernel_def = {'map':{'map_type':'linear',
                                        'map_info' : {'conditionning': args.transition_matrix_conditionning, 
                                        'bias': args.transition_bias,
                                        'range_params':args.range_transition_map_params}}, 
                                'noise_dist':Gaussian}


        emission_kernel_def = {'map':{'map_type':'nonlinear',
                                    'map_info' : {'homogeneous': True},
                                    'map': nonlinear_map_forward},
                            'noise_dist':Gaussian}


        return NonLinearHMM(args.state_dim, 
                            args.obs_dim, 
                            transition_kernel_def, 
                            emission_kernel_def, 
                            prior_dist = Gaussian,
                            num_particles = args.num_particles, 
                            num_smooth_particles=args.num_smooth_particles)
    @staticmethod
    def chaotic_rnn(args):
        nonlinear_map_forward = partial(chaotic_map, 
                                        grid_size=args.grid_size, 
                                        gamma=args.gamma,
                                        tau=args.tau)

        transition_kernel_def = {'map':{'map_type':'nonlinear',
                                        'map_info' : {'homogeneous': True},
                                        'map': nonlinear_map_forward},
                                'noise_dist':Gaussian}
        
        emission_kernel_def = {'map':{'map_type':'linear',
                                    'map_info' : {'conditionning': args.emission_matrix_conditionning, 
                                    'bias': args.emission_bias,
                                    'range_params':args.range_emission_map_params}}, 
                                'noise_dist':Student}


        return NonLinearHMM(args.state_dim, 
                            args.obs_dim, 
                            transition_kernel_def, 
                            emission_kernel_def, 
                            prior_dist = Gaussian,
                            num_particles = args.num_particles, 
                            num_smooth_particles=args.num_smooth_particles)
        
    def __init__(self, 
                state_dim, 
                obs_dim, 
                transition_kernel_def,
                emission_kernel_def,
                prior_dist = Gaussian,
                num_particles=100, 
                num_smooth_particles=None):
                                                
        HMM.__init__(self, 
                    state_dim, 
                    obs_dim, 
                    transition_kernel_type = lambda state_dim: Kernel(state_dim, state_dim, transition_kernel_def['map'], transition_kernel_def['noise_dist']), 
                    emission_kernel_type  = lambda state_dim, obs_dim:Kernel(state_dim, obs_dim, emission_kernel_def['map'], emission_kernel_def['noise_dist']),
                    prior_dist = prior_dist)

        self.smc = SMC(self.transition_kernel, 
                    self.emission_kernel, 
                    self.prior_dist, 
                    num_particles,
                    num_smooth_particles)

    def likelihood_seq(self, key, obs_seq, params):

        return self.smc.compute_filt_state_seq(key, 
                            obs_seq, 
                            self.format_params(params))[-1]
    
    def compute_filt_state_seq(self, key, obs_seq, formatted_params):

        return self.smc.compute_filt_state_seq(key, 
                                obs_seq, 
                                formatted_params)[0]

    def filt_seq(self, key, obs_seq, params):

        return self.compute_filt_state_seq(key, obs_seq, self.format_params(params))
        
    def compute_marginals(self, key, filt_seq, formatted_params):

        return self.smc.smooth_from_filt_seq(key, filt_seq, formatted_params)
    
    def smooth_seq(self, key, obs_seq, params):

        key, subkey = random.split(key, 2)

        formatted_params = self.format_params(params)

        filt_seq = self.smc.compute_filt_state_seq(key, 
                                obs_seq, 
                                formatted_params)[0]

        return self.smc.smooth_from_filt_seq(subkey, filt_seq, formatted_params)

    def filt_seq_to_mean_cov(self, key, obs_seq, params):

        weights, particles = self.filt_seq(key, obs_seq, params)
        means = vmap(lambda particles, weights: jnp.average(a=particles, axis=0, weights=weights))(particles, weights)
        covs = vmap(lambda mean, particles, weights: jnp.average(a=(particles-mean)**2, axis=0, weights=weights))(means, particles, weights)
        return means, covs 

    def smooth_seq_to_mean_cov(self, key, obs_seq, params):

        smoothing_paths = self.smooth_seq(key, obs_seq, params)
        return jnp.mean(smoothing_paths, axis=1), jnp.var(smoothing_paths, axis=1)

    def smooth_seq_at_multiple_timesteps(self, key, obs_seq, params, slices):
        key, subkey = random.split(key, 2)

        formatted_params = self.format_params(params)

        filt_seq = self.smc.compute_filt_state_seq(key, 
                                obs_seq, 
                                formatted_params)[0]

        def smooth_up_to_timestep(timestep):
            smoothing_paths = self.smc.smooth_from_filt_seq(subkey, tree_get_slice(0, timestep, filt_seq), formatted_params)
            return jnp.mean(smoothing_paths, axis=1), jnp.var(smoothing_paths, axis=1)
        means, covs = [], []

        for timestep in slices:
            mean, cov = smooth_up_to_timestep(timestep)
            means.append(mean)
            covs.append(cov)
            
        return means, covs

        
    def fit_ffbsi_em(self, key, data, optimizer, learning_rate, batch_size, num_epochs):

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
                    
                    filt_seq, logl_value = self.smc.compute_filt_state_seq(key_fwd, 
                            obs_seq, 
                            formatted_prev_theta)

                    smoothed_paths = self.smc.smooth_from_filt_seq(key_backwd, filt_seq, formatted_prev_theta)

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
            avg_logls.append(jnp.mean(avg_logl_batches))

        
        return params, avg_logls

class JohnsonBackwardSmoother(LinearBackwardSmoother):

    def __init__(self, 
                transition_kernel, 
                obs_dim, 
                update_layers, 
                explicit_proposal,
                prior_dist=Gaussian, 
                filt_dist=Gaussian):
        
        super().__init__(transition_kernel.in_dim, filt_dist)

        self.state_dim = transition_kernel.in_dim
        self.obs_dim = obs_dim
        self.explicit_proposal = explicit_proposal
        self.prior_dist:Gaussian = prior_dist
        self.transition_kernel:Kernel =  transition_kernel

        self.update_layers = update_layers
        d = self.state_dim 
        self.filt_state_shape = d + (d *(d+1)) // 2

        if explicit_proposal:
            self._format_params = lambda params: JohnsonBackwardSmootherParams(prior=self.prior_dist.format_params(params.prior), 
                                                filt_update=params.filt_update, 
                                                transition=self.transition_kernel.format_params(params.transition))

            self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(self.johnson_update_forward, 
                                                                                layers=update_layers, 
                                                                                state_dim=self.state_dim)))
            def _init_filt_state(obs, params):
                return self.filt_update_apply(params.filt_update, obs, params.prior)

            def _new_filt_state(obs, filt_state, params):

                pred_mean, pred_cov = Kalman.predict(filt_state.hidden.mean, filt_state.hidden.scale.cov, params.transition)

                pred_state = GaussianParams(mean=pred_mean, scale=Scale(cov=pred_cov))

                return self.filt_update_apply(params.filt_update, obs, pred_state)
        
        else:
            self._format_params = lambda params: JohnsonBackwardSmootherParams(prior=params.prior, 
                                                filt_update=params.filt_update, 
                                                transition=self.transition_kernel.format_params(params.transition))

            self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(self.filt_update_forward, 
                                                                                        layers=update_layers, 
                                                                                        state_dim=self.state_dim)))  
                    
            def _init_filt_state(obs, params):
                return FiltState(*self.filt_update_apply(params.filt_update, obs, params.prior))

            def _new_filt_state(obs, filt_state:FiltState, params):
                return FiltState(*self.filt_update_apply(params.filt_update, obs, filt_state.hidden))

            def _get_init_state():
                hidden_state_sizes = (*self.update_layers, self.filt_state_shape)
                return tuple([jnp.zeros(shape=[size]) for size in hidden_state_sizes])
                
            self.get_init_state = _get_init_state

        self._init_filt_state =_init_filt_state
        self._new_filt_state = _new_filt_state
                                
    def get_random_params(self, key, params_to_set=None):

        key_prior, key_transition, key_filt = random.split(key, 3)

        dummy_obs = jnp.empty((self.obs_dim,))
        transition_params = self.transition_kernel.get_random_params(key_transition)

        if self.explicit_proposal:
            prior_params = self.prior_dist.get_random_params(key_prior, 
                                                            self.state_dim)
                                                            
            dummy_pred_state = GaussianParams(mean=jnp.ones((self.state_dim,)), 
                                            scale=Scale(cov_chol=jnp.eye(self.state_dim)))
            
            filt_update_params = self.filt_update_init_params(key_filt, dummy_obs, dummy_pred_state)

        else: 

            hidden_state_sizes = (*self.update_layers, self.filt_state_shape)
            key_priors = random.split(key_prior, len(hidden_state_sizes))
            prior_params = tuple([random.normal(key, shape=[size]) for key, size in zip(key_priors, hidden_state_sizes)])
            filt_update_params = self.filt_update_init_params(key_filt, dummy_obs, prior_params)



        params =  JohnsonBackwardSmootherParams(prior_params, 
                                            transition_params, 
                                            filt_update_params)
        
        if params_to_set is not None:
            params = self.set_params(params, params_to_set)
        return params  
        
    def set_params(self, params, args):
        new_params = copy.deepcopy(params)
        for k,v in vars(args).items():         
            if k == 'default_prior_mean' and self.explicit_proposal:
                new_params.prior.mean = v * jnp.ones_like(params.prior.mean)
            elif k == 'default_prior_base_scale' and self.explicit_proposal:
                new_params.prior.scale = Scale.set_default(params.prior.scale, v, HMM.parametrization)
            elif k == 'default_transition_base_scale': 
                new_params.transition.noise.scale = Scale.set_default(params.transition.noise.scale, v, HMM.parametrization)
            # elif (k == 'default_transition_matrix') and (self.transition_kernel.map_type != 'linear'):
            #     if (type(v) == str): new_params.transition.map['linear']['w'] = jnp.load(v).astype(jnp.float64)                

        return new_params

    def init_filt_state(self, obs, params):

        return self._init_filt_state(obs, params)

    def new_filt_state(self, obs, filt_state, params):

        return self._new_filt_state(obs, filt_state, params)
    
    def format_params(self, params):
        return self._format_params(params)

    def compute_filt_state_seq(self, obs_seq, formatted_params):
        if self.explicit_proposal:
            formatted_params.compute_covs()
        return super().compute_filt_state_seq(obs_seq, formatted_params)

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params)))
        print('-- in prior + predict + backward:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.prior, params.transition))))
        print('-- in update:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params.filt_update)))
    
class GeneralBackwardSmoother(BackwardSmoother):

    def __init__(self, 
                state_dim, 
                obs_dim,
                update_layers,
                backwd_layers,
                filt_dist=Gaussian):

        self.state_dim = state_dim 
        self.obs_dim = obs_dim 

        self.update_layers = update_layers

        self.filt_state_shape = jnp.sum(jnp.array(update_layers))

        self.filt_update_init_params, self.filt_update_apply = hk.without_apply_rng(hk.transform(partial(self.filt_update_forward, 
                                                                layers=update_layers, 
                                                                state_dim=state_dim)))
        

        backwd_kernel_map_def = {'map_type':'nonlinear',
                                'map_info' : {'homogeneous': False, 'varying_params_shape':self.filt_state_shape},
                                'map': partial(self.backwd_update_forward, layers=backwd_layers)}
                

        super().__init__(filt_dist, 
                        Kernel(state_dim, state_dim, backwd_kernel_map_def, Gaussian))

    def get_random_params(self, key, args=None):
        
        key_prior, key_filt, key_back = random.split(key, 3)
    
        dummy_obs = jnp.ones((self.obs_dim,))


        key_priors = random.split(key_prior, len(self.update_layers))
        prior_params = tuple([random.normal(key, shape=[size]) for key, size in zip(key_priors, self.update_layers)])
        filt_update_params = self.filt_update_init_params(key_filt, dummy_obs, prior_params)

        backwd_params = self.backwd_kernel.get_random_params(key_back)

        return GeneralBackwardSmootherParams(prior=prior_params,
                                            filt_update=filt_update_params, 
                                            backwd=backwd_params)

    def smooth_seq(self, key, obs_seq, params, num_samples):

        formatted_params = self.format_params(params)

        filt_state_seq = self.compute_filt_state_seq(obs_seq, formatted_params)
        backwd_state_seq = self.compute_kernel_state_seq(filt_state_seq, formatted_params)

        marginals = self.compute_marginals(key, filt_state_seq, backwd_state_seq, num_samples)
        

        return jnp.mean(marginals, axis=0), jnp.var(marginals, axis=0)

    def format_params(self, params):
        return params

    def init_filt_state(self, obs, params):
        return FiltState(*self.filt_update_apply(params.filt_update, obs, params.prior))

    def new_filt_state(self, obs, filt_state:FiltState, params):
        return FiltState(*self.filt_update_apply(params.filt_update, obs, filt_state.hidden))

    def get_init_state(self):
        return tuple([jnp.zeros(shape=[size]) for size in self.update_layers])

    def new_backwd_state(self, filt_state:FiltState, params):

        return BackwardState(params.backwd, jnp.concatenate(filt_state.hidden))

    def compute_marginals(self, key, filt_state_seq, backwd_state_seq, num_samples):

        @jit
        def _sample_path(key, last_filt_state:FiltState, backwd_state_seq):
            
            keys = random.split(key, backwd_state_seq.varying.shape[0]+1)

            last_sample = self.filt_dist.sample(keys[-1], last_filt_state.out)

            def _sample_step(next_sample, x):
                
                key, backwd_state = x
                sample = self.backwd_kernel.sample(key, next_sample, backwd_state)
                return sample, sample
            
            samples = lax.scan(_sample_step, init=last_sample, xs=(keys[:-1], backwd_state_seq), reverse=True)[1]

            return tree_append(samples, last_sample)

        parallel_sampler = vmap(_sample_path, in_axes=(0,None,None))

        return parallel_sampler(random.split(key, num_samples), tree_get_idx(-1, filt_state_seq), backwd_state_seq)

    def print_num_params(self):
        params = self.get_random_params(random.PRNGKey(0))
        print('Num params:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves(params)))
        print('-- filt net:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.filt_update))))
        print('-- prior state:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.prior))))
        print('-- backwd net:', sum(jnp.atleast_1d(leaf).shape[0] for leaf in tree_leaves((params.backwd))))


        


# class NeuralLinearForwardSmoother(BackwardSmoother):

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

#         pred_mean, pred_cov = Kalman.predict(filt_state.mean, filt_state.noise.scale.cov, params.transition)
#         pred_state = vec_from_gaussian_params(GaussianParams(pred_mean, Scale(cov=pred_cov)), self.state_dim)

#         filt_state  = self.filt_update_apply(params.filt_update, obs, pred_state)

#         return gaussian_params_from_vec(filt_state, self.state_dim)

#     def new_backwd_state(self, filt_state, params):
#         raise NotImplementedError

#     def format_params(self, params):
#         return NeuralLinearForwardSmootherParams(self.prior_dist.format_params(params.prior),
#                                                 self.transition_kernel.format_params(params.transition),
#                                                 params.filt_update)





        

            
        



        

