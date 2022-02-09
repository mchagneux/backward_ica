from ast import Param
from typing import OrderedDict
import numpy as np
from utils.kalman import Kalman 
from utils.kalman import NumpyKalman
from utils.misc import * 
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from utils.linear_gaussian_hmm import LinearGaussianHMM
from utils.elbo import LinearGaussianELBO
import torch
import torch.nn as nn 
from torch.nn.utils.parametrize import register_parametrization

torch.set_default_dtype(torch.float64) 
torch.set_printoptions(precision=20)
## verbose functions 

def visualize_kalman_results(true_states, observations, filtered_state_means, filtered_state_covariances):
    
    for true_state, observation, filtered_state_mean, filtered_state_covariance in zip(true_states, observations, filtered_state_means, filtered_state_covariances): 
        plt.scatter(true_state[0], true_state[1], c='r')
        plt.scatter(observation[0], observation[1], c='b')
        plt.scatter(filtered_state_mean[0], filtered_state_mean[1], c='tab:purple')

        cov = filtered_state_covariance
        n_std = 2
        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])

        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = Ellipse((0,0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2, fill=False)

        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = filtered_state_mean[0]

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = filtered_state_mean[1]

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + plt.gca().transData)

        plt.gca().add_patch(ellipse)
        plt.show()
        plt.waitforbuttonpress()

def test_kalman():

    model_params = get_model()
    true_states, observations = sample_joint_sequence(sequence_length=20, model_params=model_params)

    filtered_state_means, filtered_state_covariances, loglikelihood = kalman_filter(observations, model_params)
    print(loglikelihood)
    filtered_state_means, filtered_state_covariances, loglikelihood = NumpyKalman(model_params).filter(observations)
    print(loglikelihood)
    visualize_kalman_results(true_states, observations, filtered_state_means, filtered_state_covariances)

def get_model():

    ### Sampling from linear Gaussian HMM 
    state_dim = 2
    obs_dim = 2

    a = 2*torch.ones(state_dim)
    A = 0.5 * torch.eye(state_dim)
    Q = torch.Tensor([[0.1,0],
                   [0,0.5]])

    B = torch.eye(obs_dim)
    b = 0*torch.ones(obs_dim)
    R = torch.Tensor([[0.2,0],
                   [0,0.3]])


    transition = Transition(matrix=A, offset=a, cov=Q)
    emission = Emission(matrix=B, offset=b, cov=R)
    prior = Prior(mean=a, cov=Q)

    return Model(transition=transition, 
                emission=emission, 
                prior=prior)

init_prior_cov = torch.diag(torch.rand(2))
init_transition_cov = torch.diag(torch.rand(2))
init_emission_cov = torch.diag(torch.rand(2))

class Diag(nn.Module):
    def forward(self, X):
        return torch.diag(X) # Return a symmetric matrix

    def right_inverse(self, A):
        return torch.diag(A)

def get_random_model():

    state_dim, obs_dim = 2, 2

    prior_mean  = nn.parameter.Parameter(torch.rand(state_dim))
    prior_cov = nn.parameter.Parameter(torch.rand((state_dim,state_dim)))

    transition_matrix = nn.parameter.Parameter(torch.diag(torch.rand(state_dim)))
    transition_offset = nn.parameter.Parameter(torch.rand(state_dim))
    transition_cov = nn.parameter.Parameter(torch.rand((state_dim,state_dim)))


    emission_matrix = nn.parameter.Parameter(torch.diag(torch.rand(obs_dim)))
    emission_offset = nn.parameter.Parameter(torch.zeros(obs_dim))
    emission_cov = nn.parameter.Parameter(torch.rand((obs_dim,obs_dim)))


    model = nn.ParameterDict({'prior_mean':prior_mean,
                        'prior_cov':prior_cov,
                        'transition_matrix':transition_matrix,
                        'transition_offset':transition_offset,
                        'transition_cov':transition_cov,
                        'emission_matrix':emission_matrix,
                        'emission_offset':emission_offset,
                        'emission_cov':emission_cov})

    register_parametrization(model,'prior_cov',Definite())
    register_parametrization(model,'transition_cov',Definite())
    register_parametrization(model,'emission_cov',Definite())

    model.prior_cov = Definite()(init_prior_cov)
    model.transition_cov =  Definite()(init_prior_cov)
    model.emission_cov =  Definite()(init_prior_cov)

    return model

# test_kalman()
model = get_random_model()

for param in model.parameters():param.requires_grad = False
states, observations = LinearGaussianHMM(model).sample_joint_sequence(10)


true_evidence = Kalman(model).filter(observations)[2]
print('True evidence:',true_evidence)
print('Difference TorchKalman and NumpyKalman:',torch.abs(true_evidence - NumpyKalman(model).filter(observations.numpy())[2]))
print('Difference log(p(x)) and ELBO when q=p:',torch.abs(true_evidence - LinearGaussianELBO(model, model)(observations)))

v_model = get_random_model()
elbo = LinearGaussianELBO(model, v_model)



optimizer = torch.optim.Adam(params=v_model.parameters(), lr=1e-2)

for _ in range(1000):
    optimizer.zero_grad()
    loss = -elbo(observations)
    loss.backward()
    optimizer.step()
    print('ELBO:', loss)








