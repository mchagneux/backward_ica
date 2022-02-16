from optax import l2_loss
from src.kalman import Kalman
from torch.nn.functional import mse_loss
import torch
class Expectation:

    @staticmethod
    def linear_additive_under_linear_gaussian(model, additive_functional, observations):
         return additive_functional(Kalman(model).smooth(observations)[0])



def mse_expectation_against_true_states(state_sequences, observation_sequences, approximate_linear_gaussian_model, additive_functional):
    
    expectations = torch.zeros((len(state_sequences),*state_sequences[0].shape[1:]))
    states_after_functional = torch.zeros((len(state_sequences),*state_sequences[0].shape[1:]))
    for sequence_nb, (states, observations) in enumerate(zip(state_sequences, observation_sequences)):

        expectation_approximate_model = Expectation.linear_additive_under_linear_gaussian(approximate_linear_gaussian_model, additive_functional, observations)
        expectations[sequence_nb] = expectation_approximate_model
        states_after_functional[sequence_nb] = additive_functional(states)
    
    return mse_loss(states_after_functional, expectations)


    