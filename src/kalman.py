from scipy.stats import multivariate_normal
import numpy as np 
from torch.distributions.multivariate_normal import MultivariateNormal
import torch 
import torch.nn as nn

class Kalman(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model 

    def predict(self, current_state_mean, current_state_covariance):
        predicted_state_mean = self.model.transition.map(current_state_mean)
        predicted_state_covariance = self.model.transition.map.weight @ current_state_covariance @ self.model.transition.map.weight.T + self.model.transition.cov
        return predicted_state_mean, predicted_state_covariance

    def update(self, predicted_state_mean, predicted_state_covariance, observation):
        predicted_observation_mean = self.model.emission.map(predicted_state_mean)
        predicted_observation_covariance = self.model.emission.map.weight @ predicted_state_covariance @ self.model.emission.map.weight.T + self.model.emission.cov
        kalman_gain = predicted_state_covariance @ self.model.emission.map.weight.T @ torch.inverse(predicted_observation_covariance)

        corrected_state_mean = predicted_state_mean + kalman_gain @ (observation - predicted_observation_mean)
        corrected_state_covariance = predicted_state_covariance - kalman_gain @ self.model.emission.map.weight @ predicted_state_covariance

        return corrected_state_mean, corrected_state_covariance

    def filter_step(self, current_state_mean, current_state_covariance, observation):
        predicted_mean, predicted_cov = self.predict(current_state_mean, current_state_covariance)
        filtered_mean, filtered_cov = self.update(predicted_mean, predicted_cov, observation)
        return predicted_mean, predicted_cov, filtered_mean, filtered_cov

    def init(self, observation):
        init_filtering_mean, init_filtering_cov = self.update(self.model.prior.mean, self.model.prior.cov, observation)
        return self.model.prior.mean, self.model.prior.cov, init_filtering_mean, init_filtering_cov

    def _log_likelihood_term(self, predicted_state_mean, predicted_state_covariance, observation):
        return MultivariateNormal(loc=self.model.emission.map(predicted_state_mean), 
                                covariance_matrix=self.model.emission.map.weight @ predicted_state_covariance @ self.model.emission.map.weight.T + self.model.emission.cov).log_prob(observation)

    def filter(self, observations):
        num_samples = len(observations)
        loglikelihood = 0
        dim_z = self.model.transition.cov.shape[0]

        filtered_state_means = torch.zeros((num_samples, dim_z), dtype=torch.float64)
        filtered_state_covariances = torch.zeros((num_samples, dim_z, dim_z), dtype=torch.float64)

        predicted_state_means = torch.zeros_like(filtered_state_means)
        predicted_state_covariances = torch.zeros_like(filtered_state_covariances)

        predicted_state_means[0], predicted_state_covariances[0], filtered_state_means[0], filtered_state_covariances[0] = self.init(observations[0])
        loglikelihood += self._log_likelihood_term(predicted_state_means[0], predicted_state_covariances[0], observations[0])
        for sample_nb in range(1, num_samples):
            predicted_state_means[sample_nb], predicted_state_covariances[sample_nb],  filtered_state_means[sample_nb],  filtered_state_covariances[sample_nb] = self.filter_step(
                                                                            filtered_state_means[sample_nb-1],
                                                                            filtered_state_covariances[sample_nb-1],
                                                                            observations[sample_nb])


            loglikelihood += self._log_likelihood_term(predicted_state_means[sample_nb], predicted_state_covariances[sample_nb], observations[sample_nb])


        return predicted_state_means, predicted_state_covariances, filtered_state_means, filtered_state_covariances, loglikelihood
    
    def smooth(self, observations):

        num_samples = len(observations)

        predicted_means, predicted_covs, filtered_means, filtered_covs, _ = self.filter(observations)

        smoothed_means, smoothed_covs = torch.zeros_like(filtered_means), torch.zeros_like(filtered_covs)

        smoothed_means[-1], smoothed_covs[-1] = filtered_means[-1], filtered_covs[-1]

        for sample_nb in reversed(range(num_samples-1)):
            A = self.model.transition.map.weight
            C = filtered_covs[sample_nb] @ A.T @ torch.inverse(predicted_covs[sample_nb+1])
            smoothed_means[sample_nb] = filtered_means[sample_nb] + C @ (smoothed_means[sample_nb+1] - predicted_means[sample_nb+1])
            smoothed_covs[sample_nb] = filtered_covs[sample_nb] + C @ (smoothed_covs[sample_nb+1] - predicted_covs[sample_nb+1]) @ C.T

        return smoothed_means, smoothed_covs



class NumpyKalman: 

    def __init__(self,
            model):

        self.transition_weight = model.transition.map.weight.numpy()
        self.transition_bias = model.transition.map.bias.numpy()
        self.transition_covariance = model.transition.cov.numpy()
        self.observation_weight = model.emission.map.weight.numpy()
        self.observation_bias = model.emission.map.bias.numpy()
        self.observation_covariance = model.emission.cov.numpy()
        self.dim_state = model.transition.cov.shape[0]
        self.prior_mean, self.prior_cov = model.prior.mean.numpy(), model.prior.cov.numpy()

    def predict(self, current_state_mean, current_state_covariance):
        predicted_state_mean = self.transition_weight @ current_state_mean + self.transition_bias
        predicted_state_covariance = self.transition_weight @ current_state_covariance @ self.transition_weight.T + self.transition_covariance
        return predicted_state_mean, predicted_state_covariance
        
    def update(self, predicted_state_mean, predicted_state_covariance, observation):
        predicted_observation_mean = self.observation_weight @ predicted_state_mean + self.observation_bias
        predicted_observation_covariance = self.observation_weight @ predicted_state_covariance @ self.observation_weight.T + self.observation_covariance
        kalman_gain = predicted_state_covariance @ self.observation_weight.T @ np.linalg.inv(predicted_observation_covariance)

        corrected_state_mean = predicted_state_mean + kalman_gain @ (observation - predicted_observation_mean)
        corrected_state_covariance = predicted_state_covariance - kalman_gain @ self.observation_weight @ predicted_state_covariance

        return corrected_state_mean, corrected_state_covariance

    def filter_step(self, current_state_mean, current_state_covariance, observation):
        predicted_mean, predicted_cov = self.predict(current_state_mean, current_state_covariance)
        filtered_mean, filtered_cov = self.update(predicted_mean, predicted_cov, observation)
        return predicted_mean, predicted_cov, filtered_mean, filtered_cov
    
    def init(self, observation):
        init_filtering_mean, init_filtering_cov = self.update(self.prior_mean, self.prior_cov, observation)
        return self.prior_mean, self.prior_cov, init_filtering_mean, init_filtering_cov

    def loglikelihood_term(self, predicted_state_mean, predicted_state_covariance, observation):
        return multivariate_normal(
            mean=self.observation_weight @ predicted_state_mean + self.observation_bias, 
            cov=self.observation_weight @ predicted_state_covariance @ self.observation_weight.T + self.observation_covariance).logpdf(observation)

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
