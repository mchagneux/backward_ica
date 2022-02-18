import torch
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization
from torch.distributions.multivariate_normal import MultivariateNormal
from abc import ABC, abstractmethod


class Diag(nn.Module):
    def forward(self, X):
        return torch.diag(X) ** 2 # Return a diagonal matrix with positive eigenvalues

    def right_inverse(self, A):
        return torch.sqrt(torch.diag(A))

class HMM(ABC):

    @staticmethod
    @abstractmethod
    def get_random_model(state_dim, obs_dim):
        pass
        
    @abstractmethod
    def sample_state_sequence(self, sequence_length):
        pass
    
    @abstractmethod
    def sample_joint_sequence(self, sequence_length):
        pass
        
class AdditiveGaussianHMM(HMM):
    def __init__(self, model=None, state_dim=2, obs_dim=2):
        if model is None: model = AdditiveGaussianHMM.get_random_model(state_dim, obs_dim) 
        self.model = model
        self.dim_z = self.model.transition.cov.shape[0]
        self.dim_x = self.model.emission.cov.shape[0]

    @staticmethod
    def get_random_model(state_dim, obs_dim):

        prior_mean  = nn.parameter.Parameter(torch.rand(state_dim))
        prior_cov = nn.parameter.Parameter(torch.rand(state_dim))
        prior = nn.ParameterDict({'mean':prior_mean,'cov':prior_cov})
        
        transition_map = nn.Linear(state_dim, state_dim, bias=True)
        transition_map.weight = nn.parameter.Parameter(torch.diag(torch.rand(state_dim)))
        transition_map.bias = nn.parameter.Parameter(torch.rand(state_dim))
        transition = nn.ModuleDict({'map':transition_map})
        transition.cov = nn.parameter.Parameter(torch.rand(state_dim))

        emission_map = nn.Sequential(nn.Linear(obs_dim, obs_dim, bias=True), nn.SELU())
        emission = nn.ModuleDict({'map':emission_map})
        emission.cov = nn.parameter.Parameter(torch.rand(obs_dim))

        
        model = nn.ModuleDict({'prior':prior, 
                            'transition':transition, 
                            'emission':emission})

        register_parametrization(model.prior,'cov', Diag())
        register_parametrization(model.transition,'cov', Diag())
        register_parametrization(model.emission,'cov', Diag())

        model.prior.cov = torch.tensor([[0.001,0],[0,0.001]])
        model.emission.cov =  torch.tensor([[0.001,0],[0,0.001]])
        model.transition.cov =  torch.tensor([[0.001,0],[0,0.001]])

        return model


    def sample_state_sequence(self, sequence_length):
        state_sequence = torch.empty(size=(sequence_length, self.dim_z))
        state_sequence[0] = MultivariateNormal(loc=self.model.prior.mean, covariance_matrix=self.model.prior.cov).sample()
        

        for sample_nb in range(1, sequence_length):
            state_sequence[sample_nb] = MultivariateNormal(loc=self.model.transition.map(state_sequence[sample_nb-1]), 
                                                            covariance_matrix=self.model.transition.cov).sample()
        
        return state_sequence

    def sample_joint_sequence(self, sequence_length):
        observation_sequence = torch.empty(size=(sequence_length, self.dim_x))
        state_sequence = self.sample_state_sequence(sequence_length)
        for sample_nb in range(sequence_length):
            observation_sequence[sample_nb] = MultivariateNormal(loc=self.model.emission.map(state_sequence[sample_nb]),
                                                                covariance_matrix=self.model.emission.cov).sample()

        return state_sequence, observation_sequence

class LinearGaussianHMM(AdditiveGaussianHMM):

    def __init__(self, model=None, state_dim=2, obs_dim=2):
        if model is None: model = LinearGaussianHMM.get_random_model(state_dim, obs_dim)
        super().__init__(model)

    @staticmethod
    def get_random_model(state_dim, obs_dim):

        prior_mean  = nn.parameter.Parameter(torch.rand(state_dim))
        prior_cov = nn.parameter.Parameter(torch.rand(state_dim))
        prior = nn.ParameterDict({'mean':prior_mean,'cov':prior_cov})
        
        transition_map = nn.Linear(state_dim, state_dim, bias=True)
        transition_map.weight = nn.parameter.Parameter(torch.diag(torch.rand(state_dim)))
        transition_map.bias = nn.parameter.Parameter(torch.rand(state_dim))
        transition = nn.ModuleDict({'map':transition_map})
        transition.cov = nn.parameter.Parameter(torch.rand(state_dim))

        emission_map = nn.Linear(obs_dim, obs_dim, bias=True)
        emission_map.weight = nn.parameter.Parameter(torch.diag(torch.rand(obs_dim)))
        emission_map.bias = nn.parameter.Parameter(torch.zeros(obs_dim))
        emission = nn.ModuleDict({'map':emission_map})
        emission.cov = nn.parameter.Parameter(torch.rand(obs_dim))

        model = nn.ModuleDict({'prior':prior, 
                            'transition':transition, 
                            'emission':emission})

        register_parametrization(model.prior,'cov', Diag())
        register_parametrization(model.transition,'cov', Diag())
        register_parametrization(model.emission,'cov', Diag())

        model.prior.cov = torch.tensor([[0.001,0],[0,0.001]])
        model.emission.cov =  torch.tensor([[0.001,0],[0,0.001]])
        model.transition.cov =  torch.tensor([[0.001,0],[0,0.001]])

        return model

                    

