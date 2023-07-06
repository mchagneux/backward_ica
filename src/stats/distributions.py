from src.utils.misc import * 
import jax.scipy.stats as stats
from jax.scipy.special import gammaln

def format_matrix(mat):
    if mat.ndim == 1:
        return jnp.diag(mat)
    return mat

@register_pytree_node_class
class Scale:
    
    parametrization = 'cov_chol'

    def __init__(self, 
                cov_chol=None, 
                prec_chol=None, 
                cov=None, 
                prec=None):

        if cov_chol is not None: 
            self._cov_chol = cov_chol

        elif prec_chol is not None: 
            self._prec_chol = prec_chol 

        elif cov is not None:
            self._cov = cov

        elif prec is not None:
            self._prec = prec


    @lazy_property
    def _cov_chol(self):
        if '_cov' in vars(self).keys():
            return cholesky(self._cov)
        return cholesky(inv(self._prec))
        
        
    @lazy_property
    def _prec_chol(self):
        if '_prec' in vars(self).keys():
            return cholesky(self._prec)
        return cholesky(inv(self._cov))


    @lazy_property
    def _cov(self):
        if '_cov_chol' in vars(self).keys():
            return mat_from_chol(self._cov_chol)
        return inv(self._prec)
    
    @lazy_property
    def _prec(self):
        if '_prec_chol' in vars(self).keys():
            return mat_from_chol(self._prec_chol)
        return inv(self._cov)

    @lazy_property
    def log_det(self):
        return log_det_from_chol(self._cov_chol)
    

    @property    
    def cov_chol(self):
        return format_matrix(self._cov_chol)
    
    @property
    def prec_chol(self):
        return format_matrix(self._prec_chol)

    @property
    def cov(self):
        return format_matrix(self._cov)
    
    @property
    def prec(self):
        return format_matrix(self._prec)


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
    def get_random(key, dim, parametrization):

        scale = random.uniform(key, shape=(dim,), minval=-1, maxval=1)

        if parametrization == 'prec_chol':scale=1/scale

        return {parametrization:scale}

    @classmethod
    def format(cls, scale):
        return cls(**scale)

    @staticmethod
    def set_default(previous_value, default_value, parametrization):
        if isinstance(previous_value, jax.Array):
            return default_value*jnp.ones_like(previous_value)
        
        scale = default_value * jnp.ones_like(previous_value[parametrization])

        if parametrization == 'prec_chol':scale=1/scale
        return {parametrization:scale}

class Gaussian: 


    @register_pytree_node_class
    class Params: 
        
        def __init__(self, mean=None, scale=None, eta1=None, eta2=None):

            if (mean is not None) and (scale is not None):
                self._mean = mean 
                self._scale = scale
            elif (eta1 is not None) and (eta2 is not None):
                self._eta1 = eta1 
                self._eta2 = eta2

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

            scale_kwargs = {Scale.parametrization:chol + chol_add(d)}
            return cls(mean=mean, scale=Scale(**scale_kwargs))
        
        @property
        def vec(self):
            return jnp.concatenate((self.eta1, jnp.diag(self.eta2).flatten()))

        @lazy_property
        def _mean(self):
            return self._scale.cov @ self._eta1

        @lazy_property
        def _scale(self):
            return Scale(prec=-2*self._eta2)
            
        @lazy_property
        def _eta1(self):
            return self._scale.prec @ self._mean
            
        @lazy_property
        def _eta2(self):
            return -0.5 * self._scale._prec
        
        @property
        def mean(self):
            return self._mean
        
        @property
        def scale(self):
            return self._scale
        
        @property
        def eta1(self):
            return self._eta1
        
        @property
        def eta2(self):
            if self._eta2.ndim == 1:
                return jnp.diag(self._eta2)
            return self._eta2
        
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
            scale_kwargs = {Scale.parametrization:chol + chol_add(d)}
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
        return stats.multivariate_normal.logpdf(x, params.mean, params.scale.cov)
    
    @staticmethod
    def pdf(x, params):
        return stats.multivariate_normal.pdf(x, params.mean, params.scale.cov)

    @classmethod
    def get_random_params(cls, key, dim):
        
        subkeys = random.split(key,2)

        mean = random.uniform(subkeys[0], shape=(dim,), minval=-1, maxval=1)
        return cls.Params(mean, Scale.get_random(key, dim, Scale.parametrization))

    @classmethod
    def format_params(cls, params):
        return cls.Params(mean=params.mean, scale=Scale.format(params.scale))

    @classmethod
    def get_random_noise_params(cls, key, dim):
        return cls.NoiseParams(Scale.get_random(key, dim, Scale.parametrization))

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
        return params.mean + jnp.diag(params.scale) @ random.t(key, params.df, 
                                                               shape=(params.mean.shape[0],))

    @staticmethod
    def logpdf(x, params):

        mean = params.mean 
        df = params.df
        scale = params.scale
        return jnp.sum(jax.vmap(jax.scipy.stats.t.logpdf, 
                                in_axes=(0, None, 0, 0))(
                                                    x, 
                                                    df, 
                                                    mean, 
                                                    scale))
    
        dim = params.mean.shape[0]
        df = params.df  
        loc = params.mean 

        dev = x - loc
        maha = dev.T @ params.scale.prec @ dev

        t = 0.5 * (df + dim)
        A = gammaln(t)
        B = gammaln(0.5 * df)
        C = dim/2. * jnp.log(df * jnp.pi)
        D = 0.5 * params.scale.log_det
        E = -t * jnp.log(1 + (1./df) * maha)

        return A - B - C - D + E

    
    @staticmethod
    def pdf(x, params):
        return jnp.exp(Student.logpdf(x, params))

    @classmethod
    def get_random_params(cls, key, dim):
        
        subkeys = random.split(key,3)


        mean = random.uniform(subkeys[0], shape=(dim,), minval=-1, maxval=1)
        df = random.randint(subkeys[1], shape=(1,), minval=1, maxval=10)
        scale = Scale.get_random(subkeys[3], dim, Scale.parametrization)[Scale.parametrization]
        return cls.Params(mean=mean, 
                            df=df, 
                            scale=scale)

    @classmethod
    def format_params(cls, params):
        return params

    @classmethod
    def get_random_noise_params(cls, key, dim):
        subkeys = random.split(key, 2)
        df = random.randint(subkeys[1], shape=(1,), minval=1, maxval=10)
        scale = Scale.get_random(subkeys[1], dim, Scale.parametrization)[Scale.parametrization]
        return cls.NoiseParams(df, scale)

    @classmethod 
    def format_noise_params(cls, noise_params):
        return noise_params #cls.NoiseParams(noise_params.df, Scale.format(noise_params.scale))
