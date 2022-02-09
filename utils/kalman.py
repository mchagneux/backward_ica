from utils.misc import * 
from utils.distributions import Gaussian
import numpy as np 
from torch.distributions.multivariate_normal import MultivariateNormal
import torch 
import torch.nn as nn


class Kalman(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model 

    def predict(self, current_state_mean, current_state_covariance):
        predicted_state_mean = self.model.transition_matrix @ current_state_mean + self.model.transition_offset
        predicted_state_covariance = self.model.transition_matrix @ current_state_covariance @ self.model.transition_matrix.T + torch.diag(self.model.transition_cov ** 2)
        return predicted_state_mean, predicted_state_covariance

    def update(self, predicted_state_mean, predicted_state_covariance, observation):
        predicted_observation_mean = self.model.emission_matrix @ predicted_state_mean + self.model.emission_offset
        predicted_observation_covariance = self.model.emission_matrix @ predicted_state_covariance @ self.model.emission_matrix.T + torch.diag(self.model.emission_cov ** 2)
        kalman_gain = predicted_state_covariance @ self.model.emission_matrix.T @ torch.inverse(predicted_observation_covariance)

        corrected_state_mean = predicted_state_mean + kalman_gain @ (observation - predicted_observation_mean)
        corrected_state_covariance = predicted_state_covariance - kalman_gain @ self.model.emission_matrix @ predicted_state_covariance

        return corrected_state_mean, corrected_state_covariance

    def filter_step(self, current_state_mean, current_state_covariance, observation):
        predicted_mean, predicted_cov = self.predict(current_state_mean, current_state_covariance)
        filtered_mean, filtered_cov = self.update(predicted_mean, predicted_cov, observation)
        return predicted_mean, predicted_cov, filtered_mean, filtered_cov

    def init(self, observation):
        init_filtering_mean, init_filtering_cov = self.update(self.model.prior_mean, torch.diag(self.model.prior_cov ** 2), observation)
        return self.model.prior_mean, torch.diag(self.model.prior_cov ** 2), init_filtering_mean, init_filtering_cov

    def _log_likelihood_term(self, predicted_state_mean, predicted_state_covariance, observation):
        return MultivariateNormal(loc=self.model.emission_matrix @ predicted_state_mean + self.model.emission_offset, 
                                covariance_matrix=self.model.emission_matrix @ predicted_state_covariance @ self.model.emission_matrix.T + torch.diag(self.model.emission_cov ** 2)).log_prob(observation)

    def filter(self, observations):
        num_samples = len(observations)
        loglikelihood = 0
        dim_z = self.model.transition_matrix.shape[0]

        filtered_state_means = torch.zeros((num_samples, dim_z), dtype=torch.float64)
        filtered_state_covariances = torch.zeros((num_samples, dim_z, dim_z), dtype=torch.float64)

        predicted_state_mean, predicted_state_covariance, filtered_state_means[0], filtered_state_covariances[0] = self.init(observations[0])
        loglikelihood += self._log_likelihood_term(predicted_state_mean, predicted_state_covariance, observations[0])
        for sample_nb in range(1, num_samples):
            predicted_state_mean, predicted_state_covariance,  filtered_state_means[sample_nb],  filtered_state_covariances[sample_nb] = self.filter_step(
                                                                            filtered_state_means[sample_nb-1],
                                                                            filtered_state_covariances[sample_nb-1],
                                                                            observations[sample_nb])


            loglikelihood += self._log_likelihood_term(predicted_state_mean, predicted_state_covariance, observations[sample_nb])


        return filtered_state_means, filtered_state_covariances, loglikelihood

class NumpyKalman: 
    def __init__(self,
            model):

        self.transition_matrix = model.transition_matrix.numpy()
        self.transition_offset = model.transition_offset.numpy()
        self.transition_covariance = np.diag(model.transition_cov.numpy() ** 2)
        self.observation_matrix = model.emission_matrix.numpy()
        self.observation_offset = model.emission_offset.numpy()
        self.observation_covariance = np.diag(model.emission_cov.numpy() ** 2)
        self.dim_state = model.transition_matrix.shape[0]
        self.prior_mean, self.prior_cov = model.prior_mean.numpy(), np.diag(model.prior_cov.numpy() ** 2)

    def predict(self, current_state_mean, current_state_covariance):
        predicted_state_mean = self.transition_matrix @ current_state_mean + self.transition_offset
        predicted_state_covariance = self.transition_matrix @ current_state_covariance @ self.transition_matrix.T + self.transition_covariance
        return predicted_state_mean, predicted_state_covariance
        
    def update(self, predicted_state_mean, predicted_state_covariance, observation):
        predicted_observation_mean = self.observation_matrix @ predicted_state_mean + self.observation_offset
        predicted_observation_covariance = self.observation_matrix @ predicted_state_covariance @ self.observation_matrix.T + self.observation_covariance
        kalman_gain = predicted_state_covariance @ self.observation_matrix.T @ np.linalg.inv(predicted_observation_covariance)

        corrected_state_mean = predicted_state_mean + kalman_gain @ (observation - predicted_observation_mean)
        corrected_state_covariance = predicted_state_covariance - kalman_gain @ self.observation_matrix @ predicted_state_covariance

        return corrected_state_mean, corrected_state_covariance

    def filter_step(self, current_state_mean, current_state_covariance, observation):
        predicted_mean, predicted_cov = self.predict(current_state_mean, current_state_covariance)
        filtered_mean, filtered_cov = self.update(predicted_mean, predicted_cov, observation)
        return predicted_mean, predicted_cov, filtered_mean, filtered_cov
    
    def init(self, observation):
        init_filtering_mean, init_filtering_cov = self.update(self.prior_mean, self.prior_cov, observation)
        return self.prior_mean, self.prior_cov, init_filtering_mean, init_filtering_cov

    def loglikelihood_term(self, predicted_state_mean, predicted_state_covariance, observation):
        return Gaussian(
            mean=self.observation_matrix @ predicted_state_mean + self.observation_offset, 
            cov=self.observation_matrix @ predicted_state_covariance @ self.observation_matrix.T + self.observation_covariance).logpdf(observation)

    def filter(self, observations):
        num_samples = len(observations)
        loglikelihood = 0

        filtered_state_means = np.zeros((num_samples, self.dim_state))
        filtered_state_covariances = np.zeros((num_samples, self.dim_state, self.dim_state))

        predicted_state_mean, predicted_state_covariance, filtered_state_means[0], filtered_state_covariances[0] = self.init(observations[0])
        loglikelihood += self.loglikelihood_term(predicted_state_mean, predicted_state_covariance, observations[0])

        for sample_nb in range(1, num_samples):
            predicted_state_mean, predicted_state_covariance, filtered_state_means[sample_nb], filtered_state_covariances[sample_nb] = self.filter_step(
                                                                            filtered_state_means[sample_nb-1],
                                                                            filtered_state_covariances[sample_nb-1],
                                                                            observations[sample_nb])

            loglikelihood += self.loglikelihood_term(predicted_state_mean, predicted_state_covariance, observations[sample_nb])


        return filtered_state_means, filtered_state_covariances, loglikelihood

