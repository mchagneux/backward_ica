from jax import numpy as jnp, random, tree_util
from .utils import _conditionnings, _mappings, Gaussian, GaussianKernel, hmm_samples
from dataclasses import dataclass
import copy 


def get_random_params(key, state_dim=2, obs_dim=2, transition_mapping_type='linear', emission_mapping_type='linear'):
    default_state_cov = 1e-3*jnp.ones(state_dim)
    default_emission_cov = 1e-3*jnp.ones(obs_dim)

    key, *subkeys = random.split(key, 2)
    prior_mean = random.uniform(subkeys[0], shape=(state_dim,))
    prior_cov = jnp.log(default_state_cov)
    prior = {'mean_params':prior_mean, 
            'cov_params':{'cov':prior_cov}}
    prior_aux = {'conditionnings':{'cov_params':{'cov':'diagonal_nonnegative'}}}

    key, *subkeys = random.split(key, 3)
    if transition_mapping_type == 'linear':
        transition_weight = random.uniform(subkeys[0], shape=(state_dim,))
        transition_bias = random.uniform(subkeys[1], shape=(state_dim,))
        transition_cov = jnp.log(default_state_cov)
        transition = {'mapping_params': {'weight':transition_weight, 'bias': transition_bias},
                    'cov_params':{'cov':transition_cov}}
        conditionnings = {'cov_params':{'cov':'diagonal_nonnegative'},
                        'mapping_params':{'weight':'diagonal'}}
        transition_aux = {'conditionnings':conditionnings, 
                            'mapping_type':transition_mapping_type}
    else: 
        raise NotImplementedError
    key, *subkeys = random.split(key, 3)
    if emission_mapping_type == 'linear':
        emission_weight = random.uniform(subkeys[0], shape=(obs_dim, state_dim))
        emission_bias = random.uniform(subkeys[1], shape=(obs_dim,))
        emission_cov = jnp.log(default_emission_cov)
        emission = {'mapping_params':{'weight':emission_weight, 'bias': emission_bias},
                    'cov_params':{'cov':emission_cov}}
        conditionnings = {'cov_params':{'cov':'diagonal_nonnegative'}}

        emission_aux = {'conditionnings':conditionnings, 
                            'mapping_type':emission_mapping_type}
    else: 
        emission_weight = random.uniform(subkeys[0], shape=(obs_dim, state_dim))
        emission_bias = random.uniform(subkeys[1], shape=(obs_dim,))
        emission_cov = jnp.log(default_emission_cov)
        emission = {'mapping_params':{'weight':emission_weight, 'bias': emission_bias},
                    'cov_params':{'cov':emission_cov}}
        conditionnings = {'cov_params':{'cov':'diagonal_nonnegative'}}

        emission_aux = {'conditionnings':conditionnings, 
                            'mapping_type':emission_mapping_type}

    raw_params = {'prior':prior, 
            'transition': transition,
            'emission':emission}

    aux = {'prior':prior_aux, 
            'transition':transition_aux, 
            'emission':emission_aux}

    return raw_params, aux

@dataclass
@tree_util.register_pytree_node_class
class GaussianHMM:

    prior: Gaussian
    transition: GaussianKernel
    emission: GaussianKernel
    
    def sample(self, key, length):
        state_key, obs_key = random.split(key, 2)
        state_keys = random.split(state_key, length)
        obs_keys = random.split(obs_key, length)
        return hmm_samples(state_keys, obs_keys, self.prior.get_sampler(), self.transition.get_sampler(), self.emission.get_sampler())

    def tree_flatten(self):
        return ((self.prior, self.transition, self.emission), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)   

    @staticmethod
    def build_from_dict(raw_params, aux):
        params = copy.deepcopy(raw_params)
        for model_part in raw_params.keys():
            conditionnings = aux[model_part]['conditionnings']
            for component_name, conditionning in conditionnings.items():
                for param_name, conditionning_type in conditionning.items():
                    params[model_part][component_name][param_name] = _conditionnings[conditionning_type](params[model_part][component_name][param_name])

        prior = Gaussian(mean=params['prior']['mean_params'], 
                        cov=params['prior']['cov_params']['cov'])

        transition = GaussianKernel(mapping=_mappings[aux['transition']['mapping_type']],
                                    mapping_params=params['transition']['mapping_params'], 
                                    cov=params['transition']['cov_params']['cov'])
                                    
        emission = GaussianKernel(mapping=_mappings[aux['transition']['mapping_type']],
                                mapping_params=params['emission']['mapping_params'],
                                cov=params['emission']['cov_params']['cov'])            
        return GaussianHMM(prior, transition, emission)


        

