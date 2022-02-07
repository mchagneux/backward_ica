from utils.misc import ModelParams, PriorParams, TransitionParams, ObservationParams
from jax.random import multivariate_normal
import jax.numpy as jnp
from jax import jit, random
from jax.numpy import dot



def sample_state_sequence(key, sequence_length, prior_params:PriorParams, transition_params:TransitionParams):
    dim_z = transition_params.matrix.shape[0]
    state_sequence = jnp.empty(shape=(sequence_length, dim_z))
    key, subkey = random.split(key)
    state_sequence = state_sequence.at[0].set(multivariate_normal(key=subkey, mean=prior_params.mean, cov=prior_params.cov))
    

    for sample_nb in range(1, sequence_length):
        key, subkey = random.split(key)
        state_sequence = state_sequence.at[sample_nb].set(multivariate_normal(key=subkey, 
                                                        mean=dot(transition_params.matrix,  state_sequence[sample_nb-1]) + transition_params.offset, 
                                                        cov=transition_params.cov))
    
    return key, state_sequence

def sample_joint_sequence(key, sequence_length, model_params:ModelParams):
    observation_params:ObservationParams = model_params.observation
    dim_x = observation_params.matrix.shape[0]
    observation_sequence = jnp.empty(shape=(sequence_length,dim_x))
    key, state_sequence = sample_state_sequence(key, sequence_length, model_params.prior, model_params.transition)
    for sample_nb in range(sequence_length):
        key, subkey = random.split(key)
        observation_sequence = observation_sequence.at[sample_nb].set(multivariate_normal(key=subkey, 
                                                                mean=dot(observation_params.matrix, state_sequence[sample_nb]) + observation_params.offset, 
                                                                cov=observation_params.cov))

    return state_sequence, observation_sequence

    

        

