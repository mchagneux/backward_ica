from src.elbo import get_appropriate_elbo
from src.kalman import Kalman
from src.hmm import LinearGaussianHMM, AdditiveGaussianHMM
import torch 


hmm = LinearGaussianHMM(state_dim=2, obs_dim=2)

# sampling 10 sequences from the hmm 
observations = hmm.sample_joint_sequence(10)[1]

true_evidence = Kalman(hmm.model).filter(observations)[-1]
elbo_linear = get_appropriate_elbo('linear_gaussian','linear_emission')




# the variational p is a random LGMM with same dimensions, and we will not learn the covariances for now
q = LinearGaussianHMM.get_random_model(2,2)
q.prior.parametrizations.cov.original.requires_grad = False
q.transition.parametrizations.cov.original.requires_grad = False 
q.emission.parametrizations.cov.original.requires_grad = False 


# elbo_nonlinear_emission = get_appropriate_elbo(variational_model_description='linear_gaussian', true_model_description='nonlinear_emission')

# elbo = elbo_nonlinear_emission(hmm.p, q)

# # print(elbo_nonlinear_emission(observation_sequences[0]))



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