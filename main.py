#%% 
from functools import partial
from src.eval import mse_expectation_against_true_states
from src.kalman import Kalman, NumpyKalman
from src.hmm import AdditiveGaussianHMM, LinearGaussianHMM
from src.elbo import LinearGaussianELBO
import torch
from tqdm import tqdm
torch.set_default_dtype(torch.float64) 
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=10)
#%%  A series of sanity checks before doing real work 

# few samples from a random LGHMM 
hmm = LinearGaussianHMM(state_dim=2, obs_dim=2)
states, observations = hmm.sample_joint_sequence(10)

for param in hmm.model.parameters():param.requires_grad = False
likelihood_torch = Kalman(hmm.model).filter(observations)[4] #kalman with torch operators 
likelihood_numpy = NumpyKalman(hmm.model).filter(observations.numpy())[2] #kalman with numpy operators 
likelihood_via_elbo = LinearGaussianELBO(hmm.model, hmm.model)(observations) #elbo

# both should be close to 0
print(likelihood_numpy - likelihood_torch)
print(likelihood_numpy - likelihood_via_elbo)
#%% Recovering parameters when p and q are linear gaussian 

hmm = LinearGaussianHMM(state_dim=2, obs_dim=2) # pick some true model p 
for param in hmm.model.parameters(): param.requires_grad = False # not learning the parameters of the true model for now 



# sampling 10 sequences from the hmm 
samples = [hmm.sample_joint_sequence(8) for _ in range(10)] 
state_sequences = [sample[0] for sample in samples]
observation_sequences = [sample[1] for sample in samples] 


# the variational model is a random LGMM with same dimensions, and we will not learn the covariances for now 
v_model = LinearGaussianHMM.get_random_model(2,2)
v_model.prior.parametrizations.cov.original.requires_grad = False
v_model.transition.parametrizations.cov.original.requires_grad = False 
v_model.emission.parametrizations.cov.original.requires_grad = False 

# the elbo object with p and q as arguments
elbo = LinearGaussianELBO(hmm.model, v_model)

# optimize the parameters of the ELBO (but theta deactivated above)
optimizer = torch.optim.Adam(params=elbo.parameters(), lr=1e-2)


true_evidence_all_sequences = sum(Kalman(hmm.model).filter(observations)[2] for observations in observation_sequences)


# optimizing model 
eps = torch.inf
while eps > 0.1:
    epoch_loss = 0.0
    for observations in observation_sequences: 
        optimizer.zero_grad()
        loss = -elbo(observations)
        loss.backward()
        optimizer.step()
        epoch_loss += -loss
    with torch.no_grad():
        eps = torch.abs(true_evidence_all_sequences - epoch_loss)
        print('Average of "L(theta, phi) - log(p_theta(x))":', eps)

# checking expectations under approximate model 
with torch.no_grad():
    additive_functional = partial(torch.sum, dim=0)
    smoothed_with_true_model = mse_expectation_against_true_states(state_sequences, observation_sequences, hmm.model, additive_functional)
    smoothed_with_approximate_model = mse_expectation_against_true_states(state_sequences, observation_sequences, v_model, additive_functional)

    print('Expectation when smooth with true model:',smoothed_with_true_model)
    print('Expectation when smooth with variational model:',smoothed_with_approximate_model)

#%% Approximating the model when p is not linear gaussian but q still is 
 
 