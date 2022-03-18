from jax import numpy as jnp, lax, config
from jax.scipy.stats.multivariate_normal import logpdf as jax_gaussian_logpdf
from pykalman.standard import KalmanFilter

from backward_ica.hmm import GaussianHMM

from .utils import * 

config.update("jax_enable_x64", True)

def predict(current_state_mean, current_state_cov, transition:GaussianKernel):
    predictive_mean = transition.map(current_state_mean)
    predictive_cov = transition.weight @ current_state_cov @ transition.weight.T + transition.cov
    return predictive_mean, predictive_cov

def update(predictive_mean, predictive_cov, observation, emission:GaussianKernel):
    predicted_observation_mean = emission.map(predictive_mean)
    predicted_observation_cov = emission.weight @ predictive_cov @ emission.weight.T + emission.cov
    kalman_gain = predictive_cov @ emission.weight.T @ jnp.linalg.inv(predicted_observation_cov)

    corrected_state_mean = predictive_mean + kalman_gain @ (observation - predicted_observation_mean)
    corrected_state_cov = predictive_cov - kalman_gain @ emission.weight @ predictive_cov

    return corrected_state_mean, corrected_state_cov

def filter_step(current_state_mean, current_state_cov, observation, transition:GaussianKernel, emission:GaussianKernel):
    predicted_mean, predicted_cov = predict(current_state_mean, current_state_cov, transition)
    filtered_mean, filtered_cov = update(predicted_mean, predicted_cov, observation, emission)
    return predicted_mean, predicted_cov, filtered_mean, filtered_cov

def init(observation, prior:Gaussian, emission:GaussianKernel):
    init_filtering_mean, init_filtering_cov = update(prior.mean, prior.cov, observation, emission)
    return prior.mean, prior.cov, init_filtering_mean, init_filtering_cov

def log_likelihood_term(predictive_mean, predictive_cov, observation, emission:GaussianKernel):
    return jax_gaussian_logpdf(x=observation, 
                        mean=emission.map(predictive_mean), 
                        cov=emission.weight @ predictive_cov @ emission.weight.T + emission.cov)


def filter(observations, hmm:GaussianHMM):
    init_predictive_mean, init_predictive_cov, init_filtering_mean, init_filtering_cov = init(observations[0], hmm.prior, hmm.emission)
    loglikelihood = log_likelihood_term(init_predictive_mean, init_predictive_cov, observations[0], hmm.emission)

    def _step(carry, x):
        loglikelihood, filtering_mean, filtering_cov, transition, emission  = carry
        predictive_mean, predictive_cov, filtering_mean, filtering_cov = filter_step(current_state_mean=filtering_mean,
                                                                                    current_state_cov=filtering_cov,
                                                                                    observation=x,
                                                                                    transition=transition,
                                                                                    emission=emission)

        loglikelihood += log_likelihood_term(predictive_mean, predictive_cov, x, emission)

        return (loglikelihood, filtering_mean, filtering_cov, transition, emission), (predictive_mean, predictive_cov, filtering_mean, filtering_cov)

    (loglikelihood, *_), (predictive_means, predictive_covs, filtering_means, filtering_covs) = lax.scan(f=_step, 
                                init=(loglikelihood, init_filtering_mean, init_filtering_cov, hmm.transition, hmm.emission), 
                                xs=observations[1:])

    predictive_means = jnp.concatenate((init_predictive_mean[None,:], predictive_means))
    predictive_covs = jnp.concatenate((init_predictive_cov[None,:], predictive_covs))
    filtering_means =  jnp.concatenate((init_filtering_mean[None,:], filtering_means))
    filtering_covs =  jnp.concatenate((init_filtering_cov[None,:], filtering_covs))

    return predictive_means, predictive_covs, filtering_means, filtering_covs, loglikelihood


def smooth_step(carry, x):
    next_smoothing_mean, next_smoothing_cov, transition_matrix = carry 
    filtering_mean, filtering_cov, next_predictive_mean, next_predictive_cov = x  
    
    C = filtering_cov @ transition_matrix @ jnp.linalg.inv(next_predictive_cov)
    smoothing_mean = filtering_mean + C @ (next_smoothing_mean - next_predictive_mean)
    smoothing_cov = filtering_cov + C @ (next_smoothing_cov - next_predictive_cov) @ C.T

    return (smoothing_mean, smoothing_cov, transition_matrix), (smoothing_mean, smoothing_cov)


def smooth(observations, hmm:GaussianHMM):

    predictive_means, predictive_covs, filtering_means, filtering_covs = filter(observations, hmm)[:-1]

    last_smoothing_mean, last_smoothing_cov = filtering_means[-1], filtering_covs[-1]

    _, (smoothing_means, smoothing_covs) = lax.scan(f=smooth_step,
                                            init=(last_smoothing_mean, last_smoothing_cov, hmm.transition.weight),
                                            xs=(filtering_means[:-1], 
                                                filtering_covs[:-1],
                                                predictive_means[1:],
                                                predictive_covs[1:]),
                                            reverse=True)

    smoothing_means = jnp.concatenate((smoothing_means, last_smoothing_mean[None,:]))
    smoothing_covs = jnp.concatenate((smoothing_covs, last_smoothing_cov[None,:]))

    return smoothing_means, smoothing_covs



def filter_pykalman(observations, hmm:GaussianHMM):

    engine = KalmanFilter(transition_matrices=hmm.transition.weight, 
                        observation_matrices=hmm.emission.weight,
                        transition_covariance=hmm.transition.cov,
                        observation_covariance=hmm.emission.cov,
                        transition_offsets=hmm.transition.bias,
                        observation_offsets=hmm.emission.bias,
                        initial_state_mean=hmm.prior.mean,
                        initial_state_covariance=hmm.prior.cov)

    return engine.filter(observations)

def smooth_pykalman(observations, hmm:GaussianHMM):
    engine = KalmanFilter(transition_matrices=hmm.transition.weight, 
                        observation_matrices=hmm.emission.weight,
                        transition_covariance=hmm.transition.cov,
                        observation_covariance=hmm.emission.cov,
                        transition_offsets=hmm.transition.bias,
                        observation_offsets=hmm.emission.bias,
                        initial_state_mean=hmm.prior.mean,
                        initial_state_covariance=hmm.prior.cov)
    return engine.smooth(observations)