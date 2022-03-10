from jax import lax, numpy as jnp, random 
from jax.nn import relu
from .misc import * 
from jax.tree_util import Partial 

def linear_map(params, input):
    return params['weight'] @ input + params['bias']

def nonlinear_map(params, input):
    return relu(params['weight'] @ input + params['bias'])

maps = {'linear':Partial(linear_map), 
        'nonlinear': Partial(nonlinear_map)}


class AdditiveGaussianHMM:

    def sample_single_step_joint(carry, x):
        key, previous_state, transition, emission = carry
        key, *subkeys = random.split(key, 3)

        new_state = random.multivariate_normal(key=subkeys[0], 
                                        mean=transition['map'](transition['params'], previous_state),
                                        cov=transition['cov'])

        new_obs = random.multivariate_normal(key=subkeys[1], 
                                    mean=emission['map'](emission['params'], new_state),
                                    cov=emission['cov'])

        return (key, new_state, transition, emission), (new_state, new_obs)

    def sample_joint_sequence(key, model, length):

        key, *subkeys = random.split(key, 3)
        init_state = random.multivariate_normal(key=subkeys[0], 
                                        mean=model['prior']['mean'],
                                        cov=model['prior']['cov'])
        emission = model['emission']
        init_obs = random.multivariate_normal(key=subkeys[1], 
                                        mean=emission['map'](emission['params'], init_state), 
                                        cov=emission['cov'])

        (key, *_), (state_samples, obs_samples) = lax.scan(f=AdditiveGaussianHMM.sample_single_step_joint, 
                                                init=(key, init_state, model['transition'], model['emission']), 
                                                length=length-1,
                                                xs=None)
        
        state_samples = jnp.concatenate((init_state[None,:], state_samples))
        obs_samples = jnp.concatenate((init_obs[None,:], obs_samples))

        return state_samples, obs_samples        

class LinearGaussianHMM:

    def get_random_model(key, state_dim=2, obs_dim=2):
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
                'transition': {'map':maps['linear'], 'params': {'weight':transition_weight, 'bias': transition_bias},'cov':transition_cov},
                'emission': {'map':maps['linear'], 'params':{'weight':emission_weight, 'bias': emission_bias},'cov':emission_cov}}



        

