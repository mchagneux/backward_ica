from torch.linalg import inv as inv 
from utils.misc import * 
from utils.distributions import Gaussian
from torch.distributions.multivariate_normal import MultivariateNormal
import torch 

def predict(current_state_mean, current_state_covariance, transition):
    predicted_state_mean = transition.matrix @ current_state_mean + transition.offset
    predicted_state_covariance = transition.matrix @ current_state_covariance @ transition.matrix.T + transition.cov
    return predicted_state_mean, predicted_state_covariance

def update(predicted_state_mean, predicted_state_covariance, observation, emission: Emission):
    predicted_observation_mean = emission.matrix @ predicted_state_mean + emission.offset
    predicted_observation_covariance = emission.matrix @ predicted_state_covariance @ emission.matrix.T + emission.cov
    kalman_gain = predicted_state_covariance  @ emission.matrix.T @ inv(predicted_observation_covariance)

    corrected_state_mean = predicted_state_mean + kalman_gain @ (observation - predicted_observation_mean)
    corrected_state_covariance = predicted_state_covariance - kalman_gain @ emission.matrix @ predicted_state_covariance

    return corrected_state_mean, corrected_state_covariance

def filter_step(current_state_mean, current_state_covariance, observation, transition:Transition, emission:Emission):
    predicted_mean, predicted_cov = predict(current_state_mean, current_state_covariance, transition)
    filtered_mean, filtered_cov = update(predicted_mean, predicted_cov, observation, emission)
    return predicted_mean, predicted_cov, filtered_mean, filtered_cov

def init(observation, prior:Prior, emission:Emission):
    init_filtering_mean, init_filtering_cov = update(prior.mean, prior.cov, observation, emission)
    return prior.mean, prior.cov, init_filtering_mean, init_filtering_cov

def log_likelihood_term(predicted_state_mean, predicted_state_covariance, observation, emission:Emission):
    return MultivariateNormal(loc=emission.matrix @ predicted_state_mean + emission.offset, 
                            covariance_matrix=emission.matrix @ predicted_state_covariance @ emission.matrix.T + emission.cov).log_prob(observation)

def filter(observations, model:Model):
    num_samples = len(observations)
    loglikelihood = 0
    dim_z = model.transition.matrix.shape[0]

    filtered_state_means = torch.zeros((num_samples, dim_z))
    filtered_state_covariances = torch.zeros((num_samples, dim_z, dim_z))

    predicted_state_mean, predicted_state_covariance, filtered_state_means[0], filtered_state_covariances[0] = init(observations[0], model.prior, model.emission)
    loglikelihood += log_likelihood_term(predicted_state_mean, predicted_state_covariance, observations[0], model.emission)

    for sample_nb in range(1, num_samples):
        predicted_state_mean, predicted_state_covariance,  filtered_state_means[sample_nb],  filtered_state_covariances[sample_nb] = filter_step(
                                                                        filtered_state_means[sample_nb-1],
                                                                        filtered_state_covariances[sample_nb-1],
                                                                        observations[sample_nb],
                                                                        model.transition,
                                                                        model.emission)


        loglikelihood += log_likelihood_term(predicted_state_mean, predicted_state_covariance, observations[sample_nb], model.emission)


    return filtered_state_means, filtered_state_covariances, loglikelihood

class Kalman: 
    def __init__(self,
            model: Model):

        self.transition_matrix = model.transition.matrix 
        self.transition_offset = model.transition.offset
        self.transition_covariance = model.transition.cov
        self.observation_matrix = model.emission.matrix
        self.observation_offset = model.emission.offset
        self.observation_covariance = model.emission.cov
        self.dim_state = model.transition.matrix.shape[0]
        self.prior_mean, self.prior_cov = model.prior.mean, model.prior.cov

    def predict(self, current_state_mean, current_state_covariance):
        predicted_state_mean = self.transition_matrix @ current_state_mean + self.transition_offset
        predicted_state_covariance = self.transition_matrix @ current_state_covariance @ self.transition_matrix.T + self.transition_covariance
        return predicted_state_mean, predicted_state_covariance
        
    def update(self, predicted_state_mean, predicted_state_covariance, observation):
        predicted_observation_mean = self.observation_matrix @ predicted_state_mean + self.observation_offset
        predicted_observation_covariance = self.observation_matrix @ predicted_state_covariance @ self.observation_matrix.T + self.observation_covariance
        kalman_gain = predicted_state_covariance @ self.observation_matrix.T @ inv(predicted_observation_covariance)

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

