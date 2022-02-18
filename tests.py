from src.elbo import get_appropriate_elbo
from src.kalman import Kalman
from src.hmm import LinearGaussianHMM, AdditiveGaussianHMM
import torch 

fully_linear_gaussian_elbo = get_appropriate_elbo('linear_gaussian','linear_emission')

hmm = AdditiveGaussianHMM(state_dim=2, obs_dim=2)

# sampling 10 sequences from the hmm 
samples = [hmm.sample_joint_sequence(8) for _ in range(10)] 
state_sequences = [sample[0] for sample in samples]
observation_sequences = [sample[1] for sample in samples] 


# the variational model is a random LGMM with same dimensions, and we will not learn the covariances for now
v_model = LinearGaussianHMM.get_random_model(2,2)
v_model.prior.parametrizations.cov.original.requires_grad = False
v_model.transition.parametrizations.cov.original.requires_grad = False 
v_model.emission.parametrizations.cov.original.requires_grad = False 


elbo_nonlinear_emission = get_appropriate_elbo(variational_model_description='linear_gaussian', true_model_description='nonlinear_emission')

elbo_nonlinear_emission = elbo_nonlinear_emission(hmm.model, v_model)

print(elbo_nonlinear_emission(observation_sequences[0]))



# # optimize the parameters of the ELBO (but theta deactivated above)
# optimizer = torch.optim.Adam(params=elbo.parameters(), lr=1e-2)


# eps = torch.inf
# # optimizing model 
# while eps > 0.1:
#     epoch_loss = 0.0
#     for observations in observation_sequences: 
#         optimizer.zero_grad()
#         loss = -elbo(observations)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += -loss
#     with torch.no_grad():
#         eps = torch.abs(true_evidence_all_sequences - epoch_loss)
#         print('Average of "L(theta, phi) - log(p_theta(x))":', eps)