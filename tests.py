import torch
from src.elbo import get_appropriate_elbo
from src.kalman import Kalman
from src.hmm import LinearGaussianHMM

hmm = LinearGaussianHMM(state_dim=2, obs_dim=2)  # we now take an hmm wih

# sampling 10 sequences from the hmm
samples = [hmm.sample_joint_sequence(8) for _ in range(10)]
state_sequences = [sample[0] for sample in samples]
observation_sequences = [sample[1] for sample in samples]
for param in hmm.model.parameters():
    param.requires_grad = False  # not learning the parameters of the true p for now


# the variational p is a random LGMM with same dimensions, and we will not learn the covariances for now
q = LinearGaussianHMM.get_random_model(2, 2)
q.prior.parametrizations.cov.original.requires_grad = False
q.transition.parametrizations.cov.original.requires_grad = False
q.emission.parametrizations.cov.original.requires_grad = False


elbo_nonlinear_emission = get_appropriate_elbo(q_description='linear_gaussian',
                                               p_description='nonlinear_emission')

elbo = elbo_nonlinear_emission(hmm.model, hmm.model)


# optimize the parameters of the ELBO (but theta deactivated above)
optimizer = torch.optim.Adam(params=elbo.parameters(), lr=1e-7)
true_evidence_all_sequences = sum(Kalman(hmm.model).filter(
observations)[-1] for observations in observation_sequences)
elbo_all_sequences = sum(elbo(observations)
                         for observations in observation_sequences)

print('True evidence accross all sequences:', true_evidence_all_sequences)
print('ELBO same model:', elbo_all_sequences-true_evidence_all_sequences)

eps = torch.inf
# optimizing p
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
