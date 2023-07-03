from .distributions import * 
import haiku as hk 
from jax import nn 
from src.utils.misc import _conditionnings
from collections import namedtuple

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
                        w_init=hk.initializers.VarianceScaling(
                                                            1.0, 
                                                            'fan_avg', 
                                                            'uniform'),
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
                            w_init=hk.initializers.VarianceScaling(
                                                                1.0, 
                                                                'fan_avg', 
                                                                'normal'))
        return x + grid_size * (-x + gamma * linear_map(nn.tanh(x))) / tau

    @staticmethod
    def linear_map_apply(map_params, input):
        out =  jnp.dot(map_params.w, input)
        return out + jnp.broadcast_to(map_params.b, out.shape)
    
    @staticmethod
    def nonlinear_map_apply(map_params, input):
        return jnp.sin(Maps.linear_map_apply(map_params, input)**2)

    @classmethod
    def linear_map_init_params(cls, key, dummy_in, out_dim, conditionning, bias, range_params):

        key_w, key_b = random.split(key, 2)

        if conditionning == 'diagonal':
            w = random.uniform(key_w, (out_dim,), minval=range_params[0], maxval=range_params[1])
        elif conditionning == 'sym_def_pos':
            d = out_dim 
            w = random.uniform(key_w, ((d*(d+1)) // 2,), minval=range_params[0], maxval=range_params[1])
        elif conditionning == 'init_sym_def_pos':
            d = out_dim 
            w = random.uniform(key_w, ((d*(d+1)) // 2,), minval=range_params[0], maxval=range_params[1])
            w = _conditionnings['sym_def_pos'](w, d)
        elif conditionning == 'init_scaled_by_dim':
            w = random.normal(key_w, (d,d)) / jnp.sqrt(d)
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



class ParametricKernel:

    Params = namedtuple('KernelParams', ['map','noise'])

    @staticmethod
    def linear_gaussian(matrix_conditonning, bias, range_params):
        transition_kernel_def = {'map_type':'linear',
                        'map_info' : {'conditionning': matrix_conditonning, 
                                    'bias': bias,
                                    'range_params':range_params}}
        return lambda in_dim, out_dim: ParametricKernel(in_dim, out_dim, transition_kernel_def)
                                                                 
    def __init__(self,
                in_dim, 
                out_dim,
                map_def, 
                noise_dist=Gaussian):

        self.in_dim = in_dim
        self.out_dim = out_dim 

        self.noise_dist = noise_dist

        self.map_type = map_def['map_type']
        self.inhomogeneous = False

        if noise_dist == Gaussian:
            self.exponential = True
            self.format_output = lambda mean, noise, params: Gaussian.Params(mean, noise.scale)
            self.params_type = Gaussian.NoiseParams
            
        elif noise_dist == Student:
            self.exponential = False
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

        elif self.map_type == 'nonlinear_analytical':
            apply_map = lambda params, input: (Maps.nonlinear_map_apply(params.map, input), params.noise)

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
                self.inhomogeneous = True
                init_map_params, nonlinear_apply_map = hk.without_apply_rng(
                                                hk.transform(partial(map_def['map'], 
                                                state_dim=out_dim)))
    
                def apply_map(params, input):
                    (mean, scale), aux = nonlinear_apply_map(params[0], params[1], input)
                    return (mean, Gaussian.NoiseParams(scale))


                get_random_map_params = lambda key: init_map_params(
                                                key, 
                                                map_def['map_info']['dummy_varying_params'], 
                                                jnp.empty((self.in_dim,)))
    
                format_map_params = lambda x:x

                self.nonlinear_map_apply = nonlinear_apply_map

        self._apply_map = apply_map 
        self._get_random_map_params = get_random_map_params
        self._format_map_params = format_map_params 
        self._get_random_noise_params = lambda key: noise_dist.get_random_noise_params(
                                                        key, 
                                                        self.out_dim)

    def map(self, x, params):
        mean, scale = self._apply_map(params, x)
        return self.format_output(mean, scale, params)
    
    def sample(self, key, x, params):
        return self.noise_dist.sample(key, self.map(x, params))

    def logpdf(self, y, x, params):
        return self.noise_dist.logpdf(y, self.map(x, params))
    
    def pdf(self, y, x, params):
        return self.noise_dist.pdf(y, self.map(x, params))
        
    def get_random_params(self, key):
        if self.inhomogeneous:
            return self._get_random_map_params(key)
        else:
            key, subkey = random.split(key, 2)
            return self.Params(self._get_random_map_params(key), 
                               self._get_random_noise_params(subkey))

    def format_params(self, params):
        return self.Params(
                        self._format_map_params(params.map), 
                        self.noise_dist.format_noise_params(params.noise))

class NonParametricKernel:
    
    Params = namedtuple('NonParametricKernelParams', ['prior', 'potential'])

    def __init__(self, prior_dist, potential_fn):
        self.prior_dist:Gaussian = prior_dist 
        self.potential_init, self.potential_apply = hk.without_apply_rng(hk.transform(potential_fn))

        def _unnorm_logpdf(x, state, params:NonParametricKernel.Params):
            return self.prior_dist.logpdf(x, params.prior) + self.potential_apply(params.potential, state, x)
        
        self._unnorm_logpdf = _unnorm_logpdf 

    def sample(self, x, params):
        raise NotImplementedError 
    
    def log_norm_cst(self, x, params, samples=None):
        log_cst = jax.scipy.special.logsumexp(jax.vmap(lambda x, y:
                        self.potential_apply(params.potential, x, y), 
                                            in_axes=(0, None))(samples, x))
        return log_cst


    def pdf(self, y, x, params, samples=None):
        return jnp.exp(self.logpdf(y, x, params, samples))

    def get_random_params(self, key):
        return self.potential_init(key)

    def format_params(self, params):
        return params 

