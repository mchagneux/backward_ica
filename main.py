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

    model.prior.cov = torch.tensor([[0.001,0],[0,0.001]])
    model.emission.cov =  torch.tensor([[0.001,0],[0,0.001]])
    model.transition.cov =  torch.tensor([[0.001,0],[0,0.001]])

    return model

# test_kalman()
model = get_random_model()

for param in model.parameters():param.requires_grad = False
hmm = LinearGaussianHMM(model)

v_model = get_random_model()
# v_model.prior.parametrizations.cov.original.requires_grad = False
# v_model.transition.parametrizations.cov.original.requires_grad = False 
# v_model.emission.parametrizations.cov.original.requires_grad = False 

elbo = LinearGaussianELBO(model, v_model)

optimizer = torch.optim.Adam(params=elbo.parameters(), lr=1e-2)

sequences = [hmm.sample_joint_sequence(8)[1] for _ in range(10)]
eps = torch.inf

true_evidence_all_sequences = sum(Kalman(model).filter(sequence)[2] for sequence in sequences)
print('True evidence accross all sequences:', true_evidence_all_sequences)

while eps > 0.1:
    epoch_loss = 0.0
    for sequence in sequences: 
        optimizer.zero_grad()
        loss = -elbo(sequence)
        loss.backward()
        optimizer.step()
        epoch_loss += -loss
    with torch.no_grad():
        eps = torch.abs(true_evidence_all_sequences - epoch_loss)
        print('Average of "L(theta, phi) - log(p_theta(x))":', eps)
    

with torch.no_grad():
    v_model = elbo.v_model
    print('Prior mean difference:', torch.abs(model.prior.mean - v_model.prior.mean).numpy())
    print('Prior cov difference:', torch.abs(model.prior.cov - v_model.prior.cov).numpy())

    print('Transition mean difference:', torch.abs(model.transition.matrix - v_model.transition.matrix).numpy())
    print('Transition offset difference:', torch.abs(model.transition.offset - v_model.transition.offset).numpy())
    print('Transition cov difference:', torch.abs(model.transition.cov - v_model.transition.cov).numpy())

    print('Emission matrix difference:', torch.abs(model.emission.matrix - v_model.emission.matrix).numpy())
    print('Emission offset difference:', torch.abs(model.emission.offset - v_model.emission.offset).numpy())
    print('Emission cov difference:', torch.abs(model.emission.cov - v_model.emission.cov).numpy())