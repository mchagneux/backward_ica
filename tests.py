from src.elbo import get_appropriate_elbo
from src.kalman import Kalman
from src.hmm import LinearGaussianHMM, AdditiveGaussianHMM
import torch 
hmm = AdditiveGaussianHMM(state_dim=2, obs_dim=2) # we now take an hmm wih 

# sampling 10 sequences from the hmm 
samples = [hmm.sample_joint_sequence(8) for _ in range(10)] 
state_sequences = [sample[0] for sample in samples]
observation_sequences = [sample[1] for sample in samples] 


# the variational p is a random LGMM with same dimensions, and we will not learn the covariances for now
q = LinearGaussianHMM.get_random_model(2,2)
q.prior.parametrizations.cov.original.requires_grad = False
q.transition.parametrizations.cov.original.requires_grad = False 
q.emission.parametrizations.cov.original.requires_grad = False 


elbo_nonlinear_emission = get_appropriate_elbo(q_description='linear_gaussian', 
                                            p_description='nonlinear_emission')

elbo = elbo_nonlinear_emission(hmm.model, q)

# print(elbo_nonlinear_emission(observation_sequences[0]))



# optimize the parameters of the ELBO (but theta deactivated above)
optimizer = torch.optim.Adam(params=elbo.parameters(), lr=1e-2)


eps = torch.inf
# optimizing p 
while True:
    epoch_loss = 0.0
    for observations in observation_sequences: 
        optimizer.zero_grad()
        loss = -elbo(observations)
        loss.backward()
        optimizer.step()
        epoch_loss += -loss
    with torch.no_grad():
        print("Loss:", epoch_loss)