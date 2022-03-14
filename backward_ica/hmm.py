from jax import lax, numpy as jnp, random, tree_util, vmap
from jax.nn import relu
from .misc import * 


def linear_map(matrix, offset, input):
    return matrix @ input + offset

def nonlinear_map(params, input):
    return relu(params['weight'] @ input + params['bias'])

mappings = {'linear':tree_util.Partial(linear_map), 
            'nonlinear': tree_util.Partial(nonlinear_map)}

# class Module1(equinox.module.Module):
#     param1:np.ndarray
#     def test(self, x):
#         return self.param1 + x
# class Module2(equinox.module.Module):
#     internal_module:Module1



# test = Module2(Module1(np.ones(2)))

# print(jax.tree_util.tree_flatten(test))

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

    def sample(self, key, state):
        return random.multivariate_normal(key=key, mean=self.map(state), cov=self.cov)

    def tree_flatten(self):
        return ((self.mapping, self.mapping_params, self.cov), self.mapping_type)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)
    

@tree_util.register_pytree_node_class
class GaussianPrior:
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov
    
    def sample(self, key):
        return random.multivariate_normal(key=key, mean=self.mean, cov=self.cov)

    def tree_flatten(self):
        return ((self.mean, self.cov), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

@tree_util.register_pytree_node_class
class HMM:
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

        def sample_state(carry, x):
            key = x
            previous_state, transition = carry

            new_state = transition.sample(key, previous_state)

            return (new_state, transition), new_state

        def sample_obs(key, emission, state):
            return 


        keys = random.split(key, 2*length)
        keys.reshape((length,2))
        init_state = self.prior.sample(key=keys[0,0])

        _, state_samples = lax.scan(f=sample_state, 
                                    init=(init_state, self.transition), 
                                    xs=keys[0,1:])
        
        state_samples = jnp.concatenate((init_state[None,:], state_samples))

        obs_samples = vmap(emission)

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
            'transition': {'mapping':'linear', 'mapping_params': {'weight':transition_weight, 'bias': transition_bias},'cov':transition_cov},
            'emission': {'mapping':'linear', 'mapping_params':{'weight':emission_weight, 'bias': emission_bias},'cov':emission_cov}}



        

