from sqlite3 import NotSupportedError
from jax import lax, numpy as jnp, random, tree_util, vmap
from jax.nn import relu
from .misc import * 


def linear_map(matrix, offset, input):
    return matrix @ input + offset

def nonlinear_map(params, input):
    return relu(params['weight'] @ input + params['bias'])

mappings = {'linear':('linear', tree_util.Partial(linear_map)), 
            'nonlinear': ('nonlinear', tree_util.Partial(nonlinear_map))}

@tree_util.register_pytree_node_class
class GaussianKernel:

    def __init__(self, mapping, mapping_params, cov, mapping_type=None):
        self.mapping = mapping
        self.mapping_params = mapping_params
        self.cov = cov
        self.mapping_type = mapping_type
        if mapping_type == "linear":
            self.weight = self.mapping_params['weight']
            self.bias = self.mapping_params['bias']
            
    def map(self, x):
        return self.mapping(self.mapping_params, x)

    def sample(self, keys, condition):
        def sample_step(key, condition, mapping, mapping_params, cov):
            return random.multivariate_normal(key=key, mean=mapping(mapping_params, condition), cov=cov)

        return vmap(sample_step, in_axes=(0, 0, None, None, None))(keys, condition, self.mapping, self.mapping_params, self.cov)

    def tree_flatten(self):
        return ((self.mapping, self.mapping_params, self.cov), self.mapping_type)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)

    def sample_chain(self, keys, init_sample):
        def chain_step(carry, x):
            previous_sample, mapping, mapping_params, cov = carry 
            key = x 
            sample = random.multivariate_normal(key=key, mean=mapping(mapping_params, previous_sample), cov=cov)
            return (sample, mapping, mapping_params, cov), sample
        return lax.scan(f=chain_step, init=(init_sample, self.mapping, self.mapping_params, self.cov), xs=keys)[1]

@tree_util.register_pytree_node_class
class GaussianPrior:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
    
    def sample(self, keys):
        def sample_step(key, mean, cov):
            return random.multivariate_normal(key=key, mean=mean, cov=cov)
        return vmap(sample_step, in_axes=(0, None, None))(keys, self.mean, self.cov)

    def tree_flatten(self):
        return ((self.mean, self.cov), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@tree_util.register_pytree_node_class
class GaussianHMM:
    def __init__(self, prior:GaussianPrior, transition:GaussianKernel, emission:GaussianKernel):
        self.prior = prior
        self.transition = transition
        self.emission = emission
        
    def tree_flatten(self):
        return ((self.prior, self.transition, self.emission))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def sample(self, key, length):

        keys = random.split(key, 2*length)
        keys.reshape((length,2))
        init_state = self.prior.sample(keys=keys[0,0])
        state_samples = jnp.concatenate((init_state[None,:], self.transition.sample_chain(keys[0,1:], init_state)))
        obs_samples = self.emission.sample(keys[1,:], condition=state_samples)

        return state_samples, obs_samples        


def get_raw_linear_gaussian_model(key, state_dim=2, obs_dim=2):
    default_state_cov = 1e-2*jnp.ones(state_dim)
    default_emission_cov = 1e-2*jnp.ones(obs_dim)

    key, *subkeys = random.split(key, 2)
    prior_mean = random.uniform(subkeys[0], shape=(state_dim,))
    prior_cov = default_state_cov

    key, *subkeys = random.split(key, 3)
    transition_weight = random.uniform(subkeys[0], shape=(state_dim,))
    transition_bias = random.uniform(subkeys[1], shape=(state_dim,))
    transition_cov = default_state_cov

    key, *subkeys = random.split(key, 3)
    emission_weight = random.uniform(subkeys[0], shape=(obs_dim,state_dim))
    emission_bias = random.uniform(subkeys[1], shape=(obs_dim,))
    emission_cov = default_emission_cov

    return {'prior':{'mean':prior_mean, 'cov':prior_cov}, 
            'transition': {'mapping':mappings['linear'], 'mapping_params': {'weight':transition_weight, 'bias': transition_bias},'cov':transition_cov},
            'emission': {'mapping':mappings['linear'], 'mapping_params':{'weight':emission_weight, 'bias': emission_bias},'cov':emission_cov}}



        

