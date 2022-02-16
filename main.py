from src.kalman import Kalman, NumpyKalman
import matplotlib.transforms as transforms
from src.hmm import AdditiveGaussianHMM, LinearGaussianHMM
from src.elbo import LinearGaussianELBO
import torch
from tqdm import tqdm
torch.set_default_dtype(torch.float64) 
torch.set_default_tensor_type(torch.DoubleTensor)
torch.set_printoptions(precision=20)


hmm = LinearGaussianHMM(state_dim=2, obs_dim=2)
states, observations = hmm.sample_joint_sequence(10)
# for param in hmm.model.named_parameters(): print(param)
for param in hmm.model.parameters():param.requires_grad = False
observations = observations[:3]
likelihood = Kalman(hmm.model).filter(observations)[2]
likelihood_numpy = NumpyKalman(hmm.model).filter(observations.numpy())[2]
likelihood_via_elbo = LinearGaussianELBO(hmm.model, hmm.model)(observations)
print(likelihood_numpy - likelihood_via_elbo)


# print(likelihood)
# print(likelihood_via_elbo)
# # example_model  = LinearGaussianHMM.get_random_model(2,2)
# # test = 0
# # for param in model.parameters() :param.requires_grad = False
# hmm = LinearGaussianHMM(state_dim=2,obs_dim=2)
# for param in hmm.model.parameters(): param.requires_grad = False

# v_model = LinearGaussianHMM.get_random_model(2,2)
# v_model.prior.parametrizations.cov.original.requires_grad = False
# v_model.transition.parametrizations.cov.original.requires_grad = False 
# v_model.emission.parametrizations.cov.original.requires_grad = False 

# elbo = LinearGaussianELBO(hmm.model, v_model)

# optimizer = torch.optim.Adam(params=elbo.parameters(), lr=1e-2)

# sequences = [hmm.sample_joint_sequence(8)[1] for _ in range(10)]
# eps = torch.inf

# # elbo_p_equals_q = LinearGaussianELBO(hmm.model, hmm.model)
# true_evidence_all_sequences = sum(Kalman(hmm.model).filter(sequence)[2] for sequence in sequences)
# # elbo_with_p_equals_q = sum(elbo_p_equals_q(sequence) for sequence in sequences)

# # print('True evidence accross all sequences:', true_evidence_all_sequences)
# # print('Elbo with p = q:', elbo_with_p_equals_q)

# while eps > 0.1:
#     epoch_loss = 0.0
#     for sequence in sequences: 
#         optimizer.zero_grad()
#         loss = -elbo(sequence)
#         loss.backward()
#         optimizer.step()
#         epoch_loss += -loss
#     with torch.no_grad():
#         eps = torch.abs(true_evidence_all_sequences - epoch_loss)
#         print('Average of "L(theta, phi) - log(p_theta(x))":', eps)
    

# with torch.no_grad():
#     v_model = elbo.v_model
#     print('Prior mean difference:', torch.abs(hmm.model.prior.mean - v_model.prior.mean).numpy())
#     print('Prior cov difference:', torch.abs(hmm.model.prior.cov - v_model.prior.cov).numpy())

#     print('Transition mean difference:', torch.abs(hmm.model.transition.map.weight - v_model.transition.map.weight).numpy())
#     print('Transition offset difference:', torch.abs(hmm.model.transition.map.bias - v_model.transition.map.bias).numpy())
#     print('Transition cov difference:', torch.abs(hmm.model.transition.cov - v_model.transition.cov).numpy())

#     print('Emission matrix difference:', torch.abs(hmm.model.emission.maps.weight - v_model.emission.maps.weight).numpy())
#     print('Emission offset difference:', torch.abs(hmm.model.emission.map.bias - v_model.emission.map.bias).numpy())
#     print('Emission cov difference:', torch.abs(hmm.model.emission.cov - v_model.emission.cov).numpy())