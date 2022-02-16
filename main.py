#%% 
from src.kalman import Kalman, NumpyKalman
import matplotlib.transforms as transforms
from src.hmm import AdditiveGaussianHMM, LinearGaussianHMM
from src.elbo import LinearGaussianELBO
import torch
from tqdm import tqdm
torch.set_default_dtype(torch.float64) 
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=20)

#%%  A series of sanity checks before doing real work 

# few samples from a random LGHMM 
hmm = LinearGaussianHMM(state_dim=2, obs_dim=2)
states, observations = hmm.sample_joint_sequence(10)

for param in hmm.model.parameters():param.requires_grad = False
likelihood_torch = Kalman(hmm.model).filter(observations)[2] #kalman with torch operators 
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
print('Summed evidence accross all sequences:', true_evidence_all_sequences)

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
    

with torch.no_grad():
    v_model = elbo.v_model
    print('Prior mean difference:', torch.abs(hmm.model.prior.mean - v_model.prior.mean).numpy())
    print('Prior cov difference:', torch.abs(hmm.model.prior.cov - v_model.prior.cov).numpy())

    print('Transition mean difference:', torch.abs(hmm.model.transition.map.weight - v_model.transition.map.weight).numpy())
    print('Transition offset difference:', torch.abs(hmm.model.transition.map.bias - v_model.transition.map.bias).numpy())
    print('Transition cov difference:', torch.abs(hmm.model.transition.cov - v_model.transition.cov).numpy())

    print('Emission matrix difference:', torch.abs(hmm.model.emission.maps.weight - v_model.emission.maps.weight).numpy())
    print('Emission offset difference:', torch.abs(hmm.model.emission.map.bias - v_model.emission.map.bias).numpy())
    print('Emission cov difference:', torch.abs(hmm.model.emission.cov - v_model.emission.cov).numpy())