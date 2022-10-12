from abc import ABCMeta, abstractmethod

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


class Maps:

    @register_pytree_node_class
    class LinearMapParams:
        def __init__(self, w, b=None):
            self.w = w 
            if b is not None: 
                self.b = b
            
        def tree_flatten(self):
            attrs = vars(self)
            children = attrs.values()
            aux_data = attrs.keys()
            return (children, aux_data)

        @classmethod
        def tree_unflatten(cls, aux_data, params):
            obj = cls.__new__(cls)
            for k,v in zip(aux_data, params):
                setattr(obj, k, v)
            return obj

        def __repr__(self):
            return str(vars(self))


    @staticmethod
    def neural_map(input, layers, slope, out_dim):

        net = hk.nets.MLP((*layers, out_dim), 
                        activate_final=True, 
                        w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                        b_init=hk.initializers.RandomNormal(),
                        activation=nn.relu)

        return net(input)
    
    @staticmethod
    def neural_map_noninjective(input, layers, slope, out_dim):

        net = hk.nets.MLP((*layers, out_dim), 
                        with_bias=False, 
                        activate_final=True, 
                        activation=nn.tanh)
        x = net(input)
        return jnp.cos(x)

    @staticmethod
    def chaotic_map(x, grid_size, gamma, tau, out_dim):
        linear_map = hk.Linear(out_dim, 
                            with_bias=False, 
                            w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'normal'))
        return x + grid_size * (-x + gamma * linear_map(nn.tanh(x))) / tau

    @staticmethod
    def linear_map_apply(map_params, input):
        out =  jnp.dot(map_params.w, input)
        return out + jnp.broadcast_to(map_params.b, out.shape)


    @classmethod
    def linear_map_init_params(cls, key, dummy_in, out_dim, conditionning, bias, range_params):

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
            return cls.LinearMapParams(w=w, b=b)
        else: 
            return cls.LinearMapParams(w=w)

    @classmethod
    def linear_map_format_params(cls, params, conditionning_func, d):

        w = conditionning_func(params.w, d)
        
        if not hasattr(params, 'b'):
            b = jnp.zeros((d,))
        else: 
            b = params.b

        return cls.LinearMapParams(w,b)

class NeuralInference: 

    @staticmethod
    def gru_net(obs, prev_state, layers):

        gru = hk.DeepRNN([hk.GRU(hidden_size) for hidden_size in (*layers,)])

        return gru(obs, prev_state)

    @staticmethod
    def gaussian_filt_net(state, d):

        out_dim = d + (d * (d + 1)) // 2
        net = hk.Linear(out_dim, 
            w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
            b_init=hk.initializers.RandomNormal(),)
    
        out = net(state.out)
        
        return Gaussian.Params.from_vec(out, d, diag=False)

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

        out = Gaussian.Params.from_vec(out, d, diag=False)

        return out.mean, out.scale

    @staticmethod
    def johnson_update_forward(obs, pred_state, layers, state_dim):

        d = state_dim 
        rec_net = hk.nets.MLP((*layers, 2*d),
                    w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                    b_init=hk.initializers.RandomNormal(),
                    activation=nn.tanh,
                    activate_final=False)


        out = rec_net(obs)
        eta1, log_prec_diag = jnp.split(out,2)

        eta2 = - 0.5 * jnp.diag(nn.softplus(log_prec_diag))

        filt_params = Gaussian.Params(eta1 = eta1 + pred_state.eta1, 
                                    eta2 = eta2 + pred_state.eta2)

        return filt_params

    @staticmethod
    def linear_gaussian_backwd_net(state, d):

        A_back_dim = d * d 
        a_back_dim = d
        Sigma_back_dim = (d * (d + 1)) // 2

        net = hk.Linear(A_back_dim + a_back_dim + Sigma_back_dim, 
                    w_init=hk.initializers.VarianceScaling(1.0, 'fan_avg', 'uniform'),
                    b_init=hk.initializers.RandomNormal(),)

        out = net(state.out)

        A_back = out[:A_back_dim].reshape((d,d))
        a_back = out[A_back_dim:A_back_dim+a_back_dim]
        Sigma_back_vec = out[A_back_dim+a_back_dim:]

        return Kernel.Params(map=Maps.LinearMapParams(w=A_back, b=a_back), 
                            noise=Gaussian.NoiseParams.from_vec(Sigma_back_vec, d))

def set_parametrization(args=None):
        
    if args is None: 
        hmm.HMM.parametrization = 'cov_chol' 
        Gaussian.Params.parametrization = 'cov_chol'


    else:
        hmm.HMM.parametrization = args.parametrization 
        Gaussian.Params.parametrization = args.parametrization

class Gaussian: 


    @register_pytree_node_class
    class Params: 

        parametrization = 'cov_chol'
        
        def __init__(self, mean=None, scale=None, eta1=None, eta2=None):

            if (mean is not None) and (scale is not None):
                self.mean = mean 
                self.scale = scale
            elif (eta1 is not None) and (eta2 is not None):
                self.eta1 = eta1 
                self.eta2 = eta2

        @classmethod
        def from_mean_scale(cls, mean, scale):
            obj = cls.__new__(cls)
            obj.mean = mean 
            obj.scale = scale
            return obj

        @classmethod
        def from_nat_params(cls, eta1, eta2):
            obj = cls.__new__(cls)
            obj.eta1 = eta1
            obj.eta2 = eta2 
            return obj

        @classmethod
        def from_vec(cls, vec, d, diag=True, chol_add=empty_add):
            mean = vec[:d]

            # def diag_chol(vec, d):
            #     return jnp.diag(vec[d:])

            # def non_diag_chol(vec, d):
            #     return chol_from_vec(vec[d:], d)
                
            if diag: 
                chol = jnp.diag(vec[d:])
            else: 
                chol = chol_from_vec(vec[d:], d)
                
            # chol = lax.cond(diag, diag_chol, non_diag_chol, vec, d)

            scale_kwargs = {cls.parametrization:chol + chol_add(d)}
            return cls(mean=mean, scale=Scale(**scale_kwargs))
        
        @property
        def vec(self):
            d = self.mean.shape[0]
            return jnp.concatenate((self.mean, self.scale.chol[jnp.tril_indices(d)]))

        @lazy_property
        def mean(self):
            return self.scale.cov @ self.eta1

        @lazy_property
        def scale(self):
            return Scale(prec=-2*self.eta2)
        
        @lazy_property
        def eta1(self):
            return self.scale.prec @ self.mean 
            
        @lazy_property
        def eta2(self):
            return -0.5 * self.scale.prec 
            
        def tree_flatten(self):
            attrs = vars(self)
            children = attrs.values()
            aux_data = attrs.keys()
            return (children, aux_data)
            
        @classmethod
        def tree_unflatten(cls, aux_data, params):
            obj = cls.__new__(cls)
            for k,v in zip(aux_data, params):
                setattr(obj, k, v)
            return obj

        def __repr__(self):
            return str(vars(self))

    @register_pytree_node_class
    @dataclass(init=True)
    class NoiseParams:
        
        scale: Scale


        @classmethod
        def from_vec(cls, vec, d, chol_add=empty_add):

            chol = chol_from_vec(vec, d)
                
            scale_kwargs = {cls.parametrization:chol + chol_add(d)}
            return cls(scale=Scale(**scale_kwargs))

        def tree_flatten(self):
            return ((self.scale,), None)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)


    @staticmethod
    def sample(key, params):
        return params.mean + params.scale.cov_chol @ random.normal(key, (params.mean.shape[0],))
    
    @staticmethod
    def logpdf(x, params):
        return gaussian_logpdf(x, params.mean, params.scale.cov)
    
    @staticmethod
    def pdf(x, params):
        return gaussian_pdf(x, params.mean, params.scale.cov)

    @classmethod
    def get_random_params(cls, key, dim):
        
        subkeys = random.split(key,2)

        mean = random.uniform(subkeys[0], shape=(dim,), minval=-1, maxval=1)
        return cls.Params(mean, Scale.get_random(key, dim, HMM.parametrization))

    @classmethod
    def format_params(cls, params):
        return cls.Params(mean=params.mean, scale=Scale.format(params.scale))

    @classmethod
    def get_random_noise_params(cls, key, dim):
        return cls.NoiseParams(Scale.get_random(key, dim, HMM.parametrization))

    @classmethod
    def format_noise_params(cls, noise_params):
        return cls.NoiseParams(Scale.format(noise_params.scale))

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


    @register_pytree_node_class
    @dataclass(init=True)
    class Params:
        
        mean: jnp.ndarray
        df: int
        scale: Scale


        def tree_flatten(self):
            return ((self.mean, self.df, self.scale), None)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

    @register_pytree_node_class
    @dataclass(init=True)
    class NoiseParams:
        
        df: int
        scale: Scale

        def tree_flatten(self):
            return ((self.df, self.scale), None)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)


    def sample(key, params):
        return params.mean + params.scale.cov_chol @ random.t(key, params.df, shape=(params.mean.shape[0],))

    @staticmethod
    def logpdf(x, params):

        return vmap(student_logpdf, in_axes=(0, None, 0, 0))(x, params.df, params.mean, jnp.diag(params.scale.cov_chol)).sum()

    
    @staticmethod
    def pdf(x, params):
        return vmap(student_pdf, in_axes=(0, None, 0, 0))(x, params.df, params.mean, jnp.diag(params.scale.cov_chol)).prod()


    @classmethod
    def get_random_params(cls, key, dim):
        
        subkeys = random.split(key,3)


        mean = random.uniform(subkeys[0], shape=(dim,), minval=-1, maxval=1)
        df = random.randint(subkeys[1], shape=(1,), minval=1, maxval=10)
        scale = Scale.get_random(subkeys[3], dim, HMM.parametrization)
        return cls.Params(mean=mean, 
                            df=df, 
                            scale=scale)

    @classmethod
    def format_params(cls, params):
        return cls.Params(mean=params.mean, df=params.df, scale=Scale.format(params.scale))

    @classmethod
    def get_random_noise_params(cls, key, dim):
        subkeys = random.split(key, 2)
        df = random.randint(subkeys[1], shape=(1,), minval=1, maxval=10)
        scale = Scale.get_random(subkeys[1], dim, HMM.parametrization)
        return cls.NoiseParams(df, scale)

    @classmethod 
    def format_noise_params(cls, noise_params):
        return cls.NoiseParams(noise_params.df, Scale.format(noise_params.scale))

class Kernel:

    Params = namedtuple('KernelParams', ['map','noise'])

    @staticmethod
    def linear_gaussian(matrix_conditonning, bias, range_params):
        transition_kernel_def = {'map_type':'linear',
                        'map_info' : {'conditionning': matrix_conditonning, 
                                    'bias': bias,
                                    'range_params':range_params}}
        return lambda in_dim, out_dim: Kernel(in_dim, out_dim, transition_kernel_def)
                                                                 
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
            self.format_output = lambda mean, noise, params: Gaussian.Params(mean, noise.scale)
            self.params_type = Gaussian.NoiseParams
            
        elif noise_dist == Student:
            self.format_output = lambda mean, noise, params: Student.Params(mean=mean, df=noise.df, scale=noise.scale)
            self.params_type = Student.NoiseParams

        if self.map_type == 'linear':

            apply_map = lambda params, input: (Maps.linear_map_apply(params.map, input), params.noise)

            init_map_params = partial(Maps.linear_map_init_params, out_dim=out_dim, 
                                    conditionning=map_def['map_info']['conditionning'], 
                                    bias=map_def['map_info']['bias'], 
                                    range_params=map_def['map_info']['range_params'])

            get_random_map_params = lambda key: init_map_params(key, jnp.empty((self.in_dim,)))

            format_map_params = partial(Maps.linear_map_format_params, 
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
                    return (mean, Gaussian.NoiseParams(scale))

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
        return self.Params(self._get_random_map_params(key), self._get_random_noise_params(subkey))

    def format_params(self, params):
        return self.Params(self._format_map_params(params.map), 
                            self.noise_dist.format_noise_params(params.noise))

class HMM: 

    @register_pytree_node_class
    @dataclass(init=True)
    class Params:
        
        prior: Gaussian.NoiseParams 
        transition: Kernel.Params
        emission: Kernel.Params

        def compute_covs(self):
            self.prior.scale.cov
            self.transition.noise.scale.cov
            self.emission.noise.scale.cov

        def tree_flatten(self):
            return ((self.prior, self.transition, self.emission), None)

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

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
        params = self.Params(prior_params, 
                        transition_params, 
                        emission_params)
        if params_to_set is not None: 
            params = self.set_params(params, params_to_set)
        return params 
        
    def format_params(self, params):

        return self.Params(self.prior_dist.format_params(params.prior),
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



    def compute_state_seq(self, obs_seq, formatted_params):

        init_state = self.init_state(obs_seq[0], 
                                    formatted_params)

        @jit
        def _step(carry, x):
            prev_state, params = carry
            obs = x
            state = self.new_state(obs, prev_state, params)
            return (state, params), state

        state_seq = lax.scan(_step, init=(init_state, formatted_params), xs=obs_seq[1:])[1]

        return tree_prepend(init_state, state_seq)

    def compute_filt_params_seq(self, state_seq, formatted_params):
        return vmap(self.filt_params_from_state, in_axes=(0,None))(state_seq, formatted_params)

    def compute_backwd_params_seq(self, state_seq, formatted_params):
        return vmap(self.backwd_params_from_state, in_axes=(0,None))(tree_droplast(state_seq), formatted_params)

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

        state_seq = self.compute_state_seq(obs_seq, formatted_params)
        filt_params_seq = self.compute_filt_params_seq(state_seq, formatted_params)
        return vmap(lambda x:x.mean)(filt_params_seq), vmap(lambda x:x.scale.cov)(filt_params_seq)
    
    def smooth_seq(self, obs_seq, params, lag=None):
        
        formatted_params = self.format_params(params)

        state_seq = self.compute_state_seq(obs_seq, formatted_params)
        filt_params_seq = self.compute_filt_params_seq(state_seq, formatted_params)
        backwd_params_seq = self.compute_backwd_params_seq(state_seq, formatted_params)

        if lag is None: 
            marginals = self.compute_marginals(tree_get_idx(-1, filt_params_seq), backwd_params_seq)
        else: 
            marginals = self.compute_fixed_lag_marginals(filt_params_seq, backwd_params_seq, lag)

        return marginals.mean, marginals.scale.cov     

    def smooth_seq_at_multiple_timesteps(self, obs_seq, params, slices):
        formatted_params = self.format_params(params)


        state_seq = self.compute_state_seq(obs_seq, formatted_params)
        filt_params_seq = self.compute_filt_params_seq(state_seq, formatted_params)
        backwd_params_seq = self.compute_backwd_params_seq(filt_params_seq, formatted_params)


        def smooth_up_to_timestep(timestep):
            marginals = self.compute_marginals(tree_get_idx(timestep, filt_params_seq), tree_get_slice(0, timestep-1, backwd_params_seq))
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

    def init_state(self, obs, params):

        mean, cov = Kalman.init(obs, params.prior, params.emission)

        return State(out=(mean, cov), hidden=None)
    
    def new_state(self, obs, prev_state, params):

        pred_mean, pred_cov = Kalman.predict(prev_state.out[0], prev_state.out[1], params.transition)
        
        mean, cov = Kalman.update(pred_mean, pred_cov, obs, params.emission)

        return State(out=(mean, cov), hidden=None)
    
    def backwd_params_from_state(self, state, params):

        return self.linear_gaussian_backwd_params_from_transition_and_filt(state.out[0], state.out[1], params.transition)

    def filt_params_from_state(self, state, params):
        return Gaussian.Params(mean=state.out[0], scale=Scale(cov=state.out[1]))

    def likelihood_seq(self, obs_seq, params):

        return Kalman.filter_seq(obs_seq, self.format_params(params))[-1]
    
    def fit_kalman_rmle(self, key, data, optimizer, learning_rate, batch_size, num_epochs, theta_star=None):
                
        
        key_init_params, key_batcher = random.split(key, 2)
        base_optimizer = getattr(optax, optimizer)(learning_rate)
        optimizer = base_optimizer
        # optimizer = optax.masked(base_optimizer, mask=HMM.Params(prior_mask, transition_mask, emission_mask))

        params = self.get_random_params(key_init_params)

        prior_scale = theta_star.prior.scale
        transition_scale = theta_star.transition.noise.scale
        emission_params = theta_star.emission

        def build_params(params):
            return HMM.Params(prior=Gaussian.Params(mean=params[0], scale=prior_scale), 
                            transition=Kernel.Params(params[1], transition_scale), 
                            emission=emission_params)
        
        params = (params.prior.mean, params.transition.map)

        loss = lambda seq, params: -self.likelihood_seq(seq, build_params(params))

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
    
    def compute_state_seq(self, obs_seq, formatted_params):
        formatted_params.compute_covs()
        return super().compute_state_seq(obs_seq, formatted_params)

class NonLinearHMM(HMM):

    @staticmethod
    def linear_transition_with_nonlinear_emission(args):
        if args.injective:
            nonlinear_map_forward = partial(Maps.neural_map, layers=args.emission_map_layers, slope=args.slope)
        else: 
            nonlinear_map_forward = partial(Maps.neural_map_noninjective, layers=args.emission_map_layers, slope=args.slope)
            
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
        nonlinear_map_forward = partial(Maps.chaotic_map, 
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

        return self.smc.compute_filt_params_seq(key, 
                            obs_seq, 
                            self.format_params(params))[-1]
    
    def compute_filt_params_seq(self, key, obs_seq, formatted_params):

        return self.smc.compute_filt_params_seq(key, 
                                obs_seq, 
                                formatted_params)[0]

    def filt_seq(self, key, obs_seq, params):

        return self.compute_filt_params_seq(key, obs_seq, self.format_params(params))
        
    def compute_marginals(self, key, filt_seq, formatted_params):

        return self.smc.smooth_from_filt_seq(key, filt_seq, formatted_params)
    
    def smooth_seq(self, key, obs_seq, params):

        key, subkey = random.split(key, 2)

        formatted_params = self.format_params(params)

        filt_seq = self.smc.compute_filt_params_seq(key, 
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

        filt_seq = self.smc.compute_filt_params_seq(key, 
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
                    
                    filt_seq, logl_value = self.smc.compute_filt_params_seq(key_fwd, 
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

        transition_kernel = hmm.Kernel.linear_gaussian(matrix_conditonning='init_invertible', 
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
        
        self._state_net = hk.without_apply_rng(hk.transform(partial(NeuralInference.gru_net, 
                                                                    layers=self.update_layers)))
        self._filt_net = hk.without_apply_rng(hk.transform(partial(NeuralInference.gaussian_filt_net, 
                                                                    d=d)))


        if self.transition_kernel is None:
            self._backwd_net = hk.without_apply_rng(hk.transform(partial(NeuralInference.linear_gaussian_backwd_net, d=d)))
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
            if k == 'default_transition_base_scale': 
                new_params.backwd.noise.scale = Scale.set_default(params.backwd.noise.scale, v, HMM.parametrization)
       
        return new_params

    def format_params(self, params):

        if self.transition_kernel is not None: 
            formatted_backwd_params = self.transition_kernel.format_params(params.backwd)

        return self.Params(params.prior, 
                            params.state, 
                            formatted_backwd_params, 
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
        


            
        



        

