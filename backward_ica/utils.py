from jax import vmap, tree_util, numpy as jnp, lax, random
from functools import partial

import tree 

## Common stateful mappings to define models, wrapped as Pytrees so that they're allowed in all jax transformations

_mappings = {'linear':tree_util.Partial(lambda params, input: params['weight'] @ input + params['bias']), 
            'nonlinear': tree_util.Partial(lambda params, input: params['weight'] @ input ** 2 + params['bias'])}


## Factory of parametrizations, e.g. for conditionned matrices  
_conditionnings = {'diagonal_nonnegative':lambda raw_param: jnp.diag(jnp.exp(raw_param)),
                'diagonal': lambda raw_param: jnp.diag(raw_param)}


def prec_and_det(cov):
    return jnp.linalg.inv(cov), jnp.linalg.det(cov)


### Useful abstractions for common stateful distributions 



@tree_util.register_pytree_node_class
class GaussianKernel:

    def __init__(self, mapping, mapping_params, cov, prec=None, det_cov=None):

        self.mapping = mapping
        self.mapping_params = mapping_params
        self.cov = cov
        self.prec = prec
        self.det_cov = det_cov

    def map(self, input):
        return self.mapping(params=self.mapping_params, input=input)

    def sample(self, key, conditioning):
        return random.multivariate_normal(key=key, mean=self.map(conditioning), cov=self.cov)
    
    def get_map(self):
        return lambda input: self.map(input)
        
    def get_sampler(self):
        return lambda key, conditioning: self.sample(key, conditioning)

    def tree_flatten(self):
        return ((self.mapping, self.mapping_params, self.cov, self.prec, self.det_cov), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@tree_util.register_pytree_node_class
class LinearGaussianKernel(GaussianKernel):

    def __init__(self, mapping, mapping_params, cov, prec=None, det_cov=None):
        super().__init__(mapping, mapping_params, cov, prec, det_cov)
        self.weight = self.mapping_params['weight']
        self.bias = self.mapping_params['bias']


@tree_util.register_pytree_node_class
class Gaussian:
    def __init__(self, mean, cov, prec=None, det_cov=None):
        self.mean = mean
        self.cov = cov
        self.prec = prec
        self.det_cov = det_cov
    
    def sample(self, key):
        return random.multivariate_normal(key=key, mean=self.mean, cov=self.cov)
    
    def get_sampler(self):
        return lambda key: random.multivariate_normal(key=key, mean=self.mean, cov=self.cov)

    def tree_flatten(self):
        return ((self.mean, self.cov, self.prec, self.det_cov), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

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