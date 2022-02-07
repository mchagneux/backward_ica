from collections import namedtuple
from jax.numpy import dot, transpose
from jax.numpy.linalg import inv
from jax import jit
import jax.numpy as jnp
from utils.misc import ModelParams, TransitionParams, ObservationParams, PriorParams

from jax.scipy.stats.multivariate_normal import logpdf as gaussian_logpdf

def predict(current_state_mean, current_state_covariance, transition_params:TransitionParams):
    predicted_state_mean = dot(transition_params.matrix, current_state_mean) + transition_params.offset
    predicted_state_covariance = dot(transition_params.matrix, current_state_covariance +  transpose(transition_params.matrix)) + transition_params.cov
    return predicted_state_mean, predicted_state_covariance
    
def update(predicted_state_mean, predicted_state_covariance, observation, observation_params: ObservationParams):
    predicted_observation_mean = dot(observation_params.matrix, predicted_state_mean) + observation_params.offset
    predicted_observation_covariance = dot(observation_params.matrix, dot(predicted_state_covariance, transpose(observation_params.matrix))) + observation_params.cov
    kalman_gain = dot(predicted_state_covariance, dot(transpose(observation_params.matrix), inv(predicted_observation_covariance)))

    corrected_state_mean = predicted_state_mean + dot(kalman_gain, (observation - predicted_observation_mean))
    corrected_state_covariance = predicted_state_covariance - dot(kalman_gain, dot( transpose(observation_params.matrix), predicted_state_covariance))

    return corrected_state_mean, corrected_state_covariance

def filter_step(current_state_mean, current_state_covariance, observation, transition_params:TransitionParams, observation_params:ObservationParams):
    predicted_mean, predicted_cov = predict(current_state_mean, current_state_covariance, transition_params)
    filtered_mean, filtered_cov = update(predicted_mean, predicted_cov, observation, observation_params)
    return predicted_mean, predicted_cov, filtered_mean, filtered_cov

def init(observation, prior_params:PriorParams, observation_params:ObservationParams):
    init_filtering_mean, init_filtering_cov = update(prior_params.mean, prior_params.cov, observation, observation_params)
    return prior_params.mean, prior_params.cov, init_filtering_mean, init_filtering_cov

def log_likelihood_term(predicted_state_mean, predicted_state_covariance, observation, observation_params:ObservationParams):
    return gaussian_logpdf(x=observation, 
                        mean=observation_params.matrix @ predicted_state_mean + observation_params.offset, 
                        cov=observation_params.matrix @ predicted_state_covariance @ observation_params.matrix.T + observation_params.cov)

def filter(observations, model_params:ModelParams):
    num_samples = len(observations)
    loglikelihood = 0

    filtered_state_means = jnp.zeros((num_samples, model_params.dims.z))
    filtered_state_covariances = jnp.zeros((num_samples, model_params.dims.z, model_params.dims.z))

    predicted_state_mean, predicted_state_covariance, filtered_state_means[0], filtered_state_covariances[0] = init(observations[0], model_params.prior, model_params.observation)
    loglikelihood += log_likelihood_term(predicted_state_mean, predicted_state_covariance, observations[0], model_params.observation)

    for sample_nb in range(1, num_samples):
        predicted_state_mean, predicted_state_covariance, filtered_state_means[sample_nb], filtered_state_covariances[sample_nb] = filter_step(
                                                                        filtered_state_means[sample_nb-1],
                                                                        filtered_state_covariances[sample_nb-1],
                                                                        observations[sample_nb],
                                                                        model_params.transition,
                                                                        model_params.observation)

        loglikelihood += log_likelihood_term(predicted_state_mean, predicted_state_covariance, observations[sample_nb], model_params.observation)


    return filtered_state_means, filtered_state_covariances, loglikelihood

            

