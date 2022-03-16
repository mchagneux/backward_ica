from jax import vmap, tree_util, nn, numpy as jnp, lax, random
from functools import partial 

## Common stateful mappings to define models, wrapped as Pytrees so that they're allowed in all jax transformations

_mappings = {'linear':(tree_util.Partial(lambda params, input: params['weight'] @ input + params['bias'])), 
            'nonlinear': (tree_util.Partial(lambda params, input: nn.relu(params['weight'] @ input + params['bias'])))}


## Factory of parametrizations, e.g. for conditionned matrices  
_conditionnings = {'diagonal_nonnegative':lambda raw_param: jnp.diag(raw_param) ** 2,
                'diagonal': lambda raw_param: jnp.diag(raw_param)}


### Useful abstractions for common stateful distributions 

@tree_util.register_pytree_node_class
class GaussianKernel:

    def __init__(self, mapping, mapping_params, cov, mapping_type=None, compute_prec_and_det=True):

        self.mapping = mapping
        self.mapping_params = mapping_params
        self.weight = None 
        self.bias = None 

        if mapping_type == "linear":
            self.weight = self.mapping_params['weight']
            self.bias = self.mapping_params['bias']

        self.cov = cov
        self.prec = None
        self.det_cov = None 
        if compute_prec_and_det: 
            self.prec = jnp.linalg.inv(cov) 
            self.det_cov = jnp.linalg.det(cov)

    def map(self, input):
        return self.mapping(params=self.mapping_params, input=input)

    def sample(self, key, conditioning):
        return random.multivariate_normal(key=key, mean=self.map(conditioning), cov=self.cov)
    
    def get_map(self):
        return lambda input: self.map(input)
        
    def get_sampler(self):
        return lambda key, conditioning: self.sample(key, conditioning)

    def tree_flatten(self):
        aux_data = [self.weight, self.bias, self.prec, self.det_cov]
        return ((self.mapping, self.mapping_params, self.cov), aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new_instance = cls(*children, None, False)
        new_instance.weight, new_instance.bias = aux_data[:2]
        new_instance.prec, new_instance.det_cov = aux_data[2:]
        return new_instance

@tree_util.register_pytree_node_class
class Gaussian:
    def __init__(self, mean, cov, compute_prec_and_det=True):
        self.mean = mean
        self.cov = cov
        self.prec = None
        self.det_cov = None
        if compute_prec_and_det:
            self.prec = jnp.linalg.inv(cov) 
            self.det_cov = jnp.linalg.det(cov)
    
    def sample(self, key):
        return random.multivariate_normal(key=key, mean=self.mean, cov=self.cov)
    
    def get_sampler(self):
        return lambda key: random.multivariate_normal(key=key, mean=self.mean, cov=self.cov)

    def tree_flatten(self):
        aux_data = [self.prec, self.det_cov]
        return ((self.mean, self.cov), aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        new_instance = cls(*children, False)
        new_instance.prec, new_instance.det_cov = aux_data
        return new_instance

### Common sampling routines 

def iid_samples(keys, sampler):
    return vmap(sampler, in_axes=0)(keys)

def conditional_samples(keys, conditional_sampler, conditionings):
    return vmap(conditional_sampler, in_axes=(0,0))(keys, conditionings) 

def conditional_samples_from_single_condition(keys, conditional_sampler, conditioning):
    return vmap(conditional_sampler, in_axes=(0,None))(keys, conditioning) 

def autoregressive_samples(keys, conditional_sampler, init_conditioning):
    def _autoregressive_sample(conditional_sampler, carry, x):
        conditioning = carry
        key = x 
        sample = conditional_sampler(key, conditioning)
        return sample, sample
    return lax.scan(f=partial(_autoregressive_sample, conditional_sampler), init=init_conditioning, xs=keys)[1]

def hmm_samples(state_keys, obs_keys, prior_sampler, state_sampler, obs_sampler):

    init_state = prior_sampler(key=state_keys[0])
    state_samples = jnp.concatenate((init_state[None,:], autoregressive_samples(state_keys[1:], state_sampler, init_state)))
    obs_samples = conditional_samples(obs_keys, obs_sampler, state_samples)
    return state_samples, obs_samples