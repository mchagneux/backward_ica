from backward_ica.utils import * 
import jax.scipy.stats as stats

@register_pytree_node_class
class Scale:

    parametrization = 'cov_chol'

    def __init__(self, cov_chol=None, prec_chol=None, cov=None, prec=None):

        if cov is not None:
            self.cov = cov
            self.cov_chol = cholesky(cov)

        elif prec is not None:
            self.prec = prec
            self.prec_chol = cholesky(prec)
        
        elif cov_chol is not None:
            self.cov_chol = cov_chol

        elif prec_chol is not None: 
            self.prec_chol = prec_chol 
        else:
            raise ValueError()        

    @lazy_property
    def cov(self):
        if 'cov_chol' in vars(self).keys():
            return mat_from_chol(self.cov_chol)
        else: return inv_from_chol(self.prec_chol)

    @lazy_property
    def prec(self):
        if 'prec_chol' in vars(self).keys():
            return mat_from_chol(self.prec_chol)
        else: return inv_from_chol(self.cov_chol)


    @lazy_property
    def cov_chol(self):
        return cholesky(self.cov)

    @lazy_property
    def prec_chol(self):
        return cholesky(self.prec)


    @property
    def chol(self):
        if 'cov_chol' in vars(self).keys(): 
            return self.cov_chol
        else: 
            return self.prec_chol

    @lazy_property
    def log_det(self):
        if 'cov_chol' in vars(self).keys():
            return log_det_from_chol(self.cov_chol)
        else: return log_det_from_chol(chol_from_prec(self.prec))


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
        base_scale =  {k:jnp.diag(v) for k,v in scale.items()}
        return cls(**base_scale)

    @staticmethod
    def set_default(previous_value, default_value, parametrization):
        scale = default_value * jnp.ones_like(previous_value[parametrization])

        if parametrization == 'prec_chol':scale=1/scale
        return {parametrization:scale}





class Gaussian: 


    @register_pytree_node_class
    class Params: 
        
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
        def from_mean_cov(cls, mean, cov):
            obj = cls.__new__(cls)
            obj.mean = mean 
            obj.scale = Scale(cov=cov)
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

            scale_kwargs = {Scale.parametrization:chol + chol_add(d)}
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
            return Scale(prec=self.eta2)
        
        @lazy_property
        def eta1(self):
            return self.scale.prec @ self.mean 
            
        @lazy_property
        def eta2(self):
            return self.scale.prec 
            
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
        return params.mean + params.scale.cov_chol @ random.t(key, params.df, shape=(params.mean.shape[0],))

    @staticmethod
    def logpdf(x, params):

        return vmap(stats.t.logpdf, in_axes=(0, None, 0, 0))(x, params.df, params.mean, jnp.diag(params.scale.cov_chol)).sum()

    
    @staticmethod
    def pdf(x, params):
        return vmap(stats.t.pdf, in_axes=(0, None, 0, 0))(x, params.df, params.mean, jnp.diag(params.scale.cov_chol)).prod()


    @classmethod
    def get_random_params(cls, key, dim):
        
        subkeys = random.split(key,3)


        mean = random.uniform(subkeys[0], shape=(dim,), minval=-1, maxval=1)
        df = random.randint(subkeys[1], shape=(1,), minval=1, maxval=10)
        scale = Scale.get_random(subkeys[3], dim, Scale.parametrization)
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
        scale = Scale.get_random(subkeys[1], dim, Scale.parametrization)
        return cls.NoiseParams(df, scale)

    @classmethod 
    def format_noise_params(cls, noise_params):
        return cls.NoiseParams(noise_params.df, Scale.format(noise_params.scale))
