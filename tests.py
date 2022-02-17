from src.elbo import LinearGaussianELBO

from src.hmm import LinearGaussianHMM

hmm = LinearGaussianHMM(state_dim=2, obs_dim=2)
v_model = LinearGaussianHMM.get_random_model(2,2)

observations = hmm.sample_joint_sequence(10)[1]


elbo = LinearGaussianELBO(hmm.model, v_model)


test = elbo(observations)


