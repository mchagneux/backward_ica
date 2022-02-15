from utils.kalman import Kalman 
from utils.kalman import NumpyKalman
from utils.misc import * 
import matplotlib.transforms as transforms
from utils.hmm import LinearGaussianHMM
from utils.elbo import LinearGaussianELBO
import torch
import torch.nn as nn 
from torch.nn.utils.parametrize import register_parametrization
from tqdm import tqdm
torch.set_default_dtype(torch.float64) 
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=20)




state_dim, obs_dim = 2, 2

class Diag(nn.Module):
    def forward(self, X):
        return torch.diag(X) ** 2 # Return a diagonal matrix with positive eigenvalues

    def right_inverse(self, A):
        return torch.sqrt(torch.diag(A))

def get_random_model():


    prior_mean  = nn.parameter.Parameter(torch.rand(state_dim))
    prior_cov = nn.parameter.Parameter(torch.rand(state_dim))
    prior = nn.ParameterDict({'mean':prior_mean,'cov':prior_cov})
    

    transition_matrix = nn.parameter.Parameter(torch.diag(torch.rand(state_dim)))
    transition_offset = nn.parameter.Parameter(torch.rand(state_dim))
    transition_cov = nn.parameter.Parameter(torch.rand(state_dim))
    transition = nn.ParameterDict({'matrix':transition_matrix,'offset':transition_offset, 'cov':transition_cov})

    emission_matrix = nn.parameter.Parameter(torch.diag(torch.rand(obs_dim)))
    emission_offset = nn.parameter.Parameter(torch.zeros(obs_dim))
    emission_cov = nn.parameter.Parameter(torch.rand(obs_dim))
    emission = nn.ParameterDict({'matrix':emission_matrix,'offset':emission_offset, 'cov':emission_cov})

    model = nn.ModuleDict({'prior':prior, 
                        'transition':transition, 
                        'emission':emission})

    register_parametrization(model.prior,'cov', Diag())
    register_parametrization(model.transition,'cov', Diag())
    register_parametrization(model.emission,'cov', Diag())

    model.prior.cov = torch.tensor([[0.01,0],[0,0.01]])
    model.emission.cov =  torch.tensor([[0.01,0],[0,0.01]])
    model.transition.cov =  torch.tensor([[0.01,0],[0,0.01]])

    return model

# test_kalman()
model = get_random_model()

for param in model.parameters():param.requires_grad = False
hmm = LinearGaussianHMM(model)
states, observations = hmm.sample_joint_sequence(100)


true_evidence = Kalman(model).filter(observations)[2]
print('True evidence:',true_evidence)
print('Difference TorchKalman and NumpyKalman:',torch.abs(true_evidence - NumpyKalman(model).filter(observations.numpy())[2]))
print('Difference log(p(x)) and ELBO when q=p:',torch.abs(true_evidence - LinearGaussianELBO(model, model)(observations)))


elbo = LinearGaussianELBO(model, get_random_model())


optimizer = torch.optim.Adam(params=elbo.parameters(), lr=1e-2)

eps = torch.inf

while eps > 1:
    optimizer.zero_grad()
    loss = -elbo(observations)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        eps = torch.abs(true_evidence + loss)
        print('L(theta, phi) - log(p_theta(x)):', eps)
    

for model_param, v_model_param in zip(elbo.model, elbo.v_model):
    print(torch.abs(model_param - v_model_param))
