from utils.kalman import Kalman 
from utils.kalman import NumpyKalman
from utils.misc import * 
import matplotlib.transforms as transforms
from utils.linear_gaussian_hmm import LinearGaussianHMM
from utils.elbo import LinearGaussianELBO
import torch
import torch.nn as nn 
from torch.nn.utils.parametrize import register_parametrization
from tqdm import tqdm
torch.set_default_dtype(torch.float64) 
torch.set_printoptions(precision=20)
## verbose functions 




state_dim, obs_dim = 2, 2


# class Diag(nn.Module):
#     def forward(self, X):
#         return torch.diag(X) # Return a symmetric matrix

#     def right_inverse(self, A):
#         return torch.diag(A)

def get_random_model():


    prior_mean  = nn.parameter.Parameter(torch.rand(state_dim))
    prior_cov = nn.parameter.Parameter(torch.tensor([0.1, 0.1], dtype=torch.float64))
    

    transition_matrix = nn.parameter.Parameter(torch.diag(torch.rand(state_dim)))
    transition_offset = nn.parameter.Parameter(torch.rand(state_dim))
    transition_cov = nn.parameter.Parameter(torch.tensor([0.1, 0.1], dtype=torch.float64))


    emission_matrix = nn.parameter.Parameter(torch.diag(torch.rand(obs_dim)))
    emission_offset = nn.parameter.Parameter(torch.zeros(obs_dim))
    emission_cov = nn.parameter.Parameter(torch.tensor([0.1, 0.1], dtype=torch.float64))

    model = nn.ParameterDict({'prior_mean':prior_mean,
                        'prior_cov':prior_cov,
                        'transition_matrix':transition_matrix,
                        'transition_offset':transition_offset,
                        'transition_cov':transition_cov,
                        'emission_matrix':emission_matrix,
                        'emission_offset':emission_offset,
                        'emission_cov':emission_cov})

    # register_parametrization(model,'prior_cov',Diag(),unsafe=True)
    # register_parametrization(model,'transition_cov',Diag(), unsafe=True)
    # register_parametrization(model,'emission_cov',Diag(), unsafe=True)

    # model.prior_cov = init_prior_cov
    # model.transition_cov =  init_prior_cov
    # model.emission_cov =  init_prior_cov

    return model

# test_kalman()
model = get_random_model()

for param in model.parameters():param.requires_grad = False
hmm = LinearGaussianHMM(model)
# states, observations = hmm.sample_joint_sequence(30)


# true_evidence = Kalman(model).filter(observations)[2]
# print('True evidence:',true_evidence)
# print('Difference TorchKalman and NumpyKalman:',torch.abs(true_evidence - NumpyKalman(model).filter(observations.numpy())[2]))
# print('Difference log(p(x)) and ELBO when q=p:',torch.abs(true_evidence - LinearGaussianELBO(model, model)(observations)))

v_model = get_random_model()
# v_model.prior_cov.requires_grad = False 
# v_model.transition_cov.requires_grad = False 
# v_model.emission_cov.requires_grad = False

elbo = LinearGaussianELBO(model, v_model)
# scipted_elbo = torch.jit.script(elbo)

# print(elbo(observations))
# print(scipted_elbo(observations))
optimizer = torch.optim.Adam(params=elbo.parameters(), lr=1e-2)

sequences = [hmm.sample_joint_sequence(8)[1] for _ in range(15)]
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
    print('Prior mean difference:', torch.abs(model.prior_mean - v_model.prior_mean).numpy())
    print('Prior cov difference:', torch.abs(model.prior_cov**2 - v_model.prior_cov**2).numpy())

    print('Transition mean difference:', torch.abs(model.transition_matrix - v_model.transition_matrix).numpy())
    print('Transition offset difference:', torch.abs(model.transition_offset - v_model.transition_offset).numpy())
    print('Transition cov difference:', torch.abs(model.transition_cov**2 - v_model.transition_cov**2).numpy())

    print('Emission matrix difference:', torch.abs(model.emission_matrix - v_model.emission_matrix).numpy())
    print('Emission offset difference:', torch.abs(model.emission_offset - v_model.emission_offset).numpy())
    print('Emission cov difference:', torch.abs(model.emission_cov**2 - v_model.emission_cov**2).numpy())