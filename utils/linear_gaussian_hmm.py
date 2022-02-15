from utils.misc import * 
import torch
from torch.distributions.multivariate_normal import MultivariateNormal


class LinearGaussianHMM:

    def __init__(self, model):
        self.model = model 

    def sample_state_sequence(self, sequence_length):
        dim_z = self.model.transition_matrix.shape[0]
        state_sequence = torch.empty(size=(sequence_length, dim_z))
        state_sequence[0] = MultivariateNormal(loc=self.model.prior_mean, covariance_matrix=torch.diag(self.model.prior_cov ** 2)).sample()
        

        for sample_nb in range(1, sequence_length):
            state_sequence[sample_nb] = MultivariateNormal(loc=self.model.transition_matrix @ state_sequence[sample_nb-1] + self.model.transition_offset, 
                                                            covariance_matrix=torch.diag(self.model.transition_cov ** 2)).sample()
        
        return state_sequence

    def sample_joint_sequence(self, sequence_length):
        dim_x = self.model.emission_matrix.shape[0]
        observation_sequence = torch.empty(size=(sequence_length,dim_x))
        state_sequence = self.sample_state_sequence(sequence_length)
        for sample_nb in range(sequence_length):
            observation_sequence[sample_nb] = MultivariateNormal(loc=self.model.emission_matrix @ state_sequence[sample_nb] + self.model.emission_offset, 
                                                                covariance_matrix=torch.diag(self.model.emission_cov ** 2)).sample()

        return state_sequence, observation_sequence

                    

