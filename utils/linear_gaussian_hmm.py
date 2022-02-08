from utils.misc import * 
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


def sample_state_sequence(sequence_length, prior_params:Prior, transition_params:Transition):
    dim_z = transition_params.matrix.shape[0]
    state_sequence = torch.empty(shape=(sequence_length, dim_z))
    state_sequence[0] = MultivariateNormal(loc=prior_params.mean, covariance_matrix=prior_params.cov).sample()
    

    for sample_nb in range(1, sequence_length):
        state_sequence[sample_nb] = MultivariateNormal(loc=transition_params.matrix @ state_sequence[sample_nb-1] + transition_params.offset, 
                                                        covariance_matrix=transition_params.cov).sample()
    
    return state_sequence

def sample_joint_sequence(sequence_length, model_params:Model):
    observation_params:Emission = model_params.emission
    dim_x = observation_params.matrix.shape[0]
    observation_sequence = torch.empty(size=(sequence_length,dim_x))
    key, state_sequence = sample_state_sequence(sequence_length, model_params.prior, model_params.transition)
    for sample_nb in range(sequence_length):
        observation_sequence[sample_nb] = MultivariateNormal(loc=observation_params.matrix @ state_sequence[sample_nb] + observation_params.offset, 
                                                            covariance_matrix=observation_params.cov).sample()

    return state_sequence, observation_sequence

    

        

