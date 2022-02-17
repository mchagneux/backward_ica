from src.elbo import QLinearGaussian
from src.kalman import Kalman
from src.hmm import LinearGaussianHMM
import torch 
hmm = LinearGaussianHMM(state_dim=2, obs_dim=2)
for param in hmm.model.parameters(): param.requires_grad = False
observations = hmm.sample_joint_sequence(10)[1]


with torch.no_grad():
    print(torch.abs(QLinearGaussian(hmm.model, hmm.model)(observations)  - Kalman(hmm.model).filter(observations)[-1]))


