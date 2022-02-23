from jax.numpy.linalg import inv
import jax.numpy as jnp
import numpy as np
import jax
from .misc import * 
from jax.scipy.stats.multivariate_normal import logpdf as gaussian_logpdf
from scipy.stats import multivariate_normal
jax.config.update("jax_enable_x64", True)

def predict(current_state_mean, current_state_cov, transition):
    predictive_mean = transition.weight @ current_state_mean + transition.bias
    predictive_cov = transition.weight @ current_state_cov @ transition.weight.T + transition.cov
    return predictive_mean, predictive_cov

def update(predictive_mean, predictive_cov, observation, emission: Emission):
    predicted_observation_mean = emission.weight @ predictive_mean + emission.bias
    predicted_observation_cov = emission.weight @ predictive_cov @ emission.weight.T + emission.cov
    kalman_gain = predictive_cov  @ emission.weight.T @ inv(predicted_observation_cov)

    corrected_state_mean = predictive_mean + kalman_gain @ (observation - predicted_observation_mean)
    corrected_state_cov = predictive_cov - kalman_gain @ emission.weight @ predictive_cov

    return corrected_state_mean, corrected_state_cov

def filter_step(current_state_mean, current_state_cov, observation, transition:Transition, emission:Emission):
    predicted_mean, predicted_cov = predict(current_state_mean, current_state_cov, transition)
    filtered_mean, filtered_cov = update(predicted_mean, predicted_cov, observation, emission)
    return predicted_mean, predicted_cov, filtered_mean, filtered_cov

def init(observation, prior:Prior, emission:Emission):
    init_filtering_mean, init_filtering_cov = update(prior.mean, prior.cov, observation, emission)
    return prior.mean, prior.cov, init_filtering_mean, init_filtering_cov

def log_likelihood_term(predictive_mean, predictive_cov, observation, emission:Emission):
    return gaussian_logpdf(x=observation, 
                        mean=emission.weight @ predictive_mean + emission.bias, 
                        cov=emission.weight @ predictive_cov @ emission.weight.T + emission.cov)


def filter(observations, model:Model):
    init_predictive_mean, init_predictive_cov, init_filtering_mean, init_filtering_cov = init(observations[0], model.prior, model.emission)
    loglikelihood = log_likelihood_term(init_predictive_mean, init_predictive_cov, observations[0], model.emission)

    def _step(carry, x):
        loglikelihood, filtering_mean, filtering_cov, transition, emission  = carry
        predictive_mean, predictive_cov, filtering_mean, filtering_cov = filter_step(current_state_mean=filtering_mean,
                                                                                    current_state_cov=filtering_cov,
                                                                                    observation=x,
                                                                                    transition=transition,
                                                                                    emission=emission)

        loglikelihood += log_likelihood_term(predictive_mean, predictive_cov, x, emission)

        return (loglikelihood, filtering_mean, filtering_cov, transition, emission), (predictive_mean, predictive_cov, filtering_mean, filtering_cov)

    (loglikelihood, *_), (predictive_means, predictive_covs, filtering_means, filtering_covs) = jax.lax.scan(f=_step, 
                                init=(loglikelihood, init_filtering_mean, init_filtering_cov, model.transition, model.emission), 
                                xs=observations[1:])

    predictive_means = jnp.concatenate((init_predictive_mean[None,:], predictive_means))
    predictive_covs = jnp.concatenate((init_predictive_cov[None,:], predictive_covs))
    filtering_means =  jnp.concatenate((init_filtering_mean[None,:], filtering_means))
    filtering_covs =  jnp.concatenate((init_filtering_cov[None,:], filtering_covs))

    return predictive_means, predictive_covs, filtering_means, filtering_covs, loglikelihood

class NumpyKalman: 

    def __init__(self,
            model:Model):

        self.transition_weight = np.array(model.transition.weight)
        self.transition_bias = np.array(model.transition.bias)
        self.transition_cov =  np.array(model.transition.cov)
        self.observation_weight = np.array(model.emission.weight)
        self.observation_bias = np.array(model.emission.bias)
        self.observation_cov = np.array(model.emission.cov)
        self.dim_state = model.transition.cov.shape[0]
        self.prior_mean, self.prior_cov = np.array(model.prior.mean), np.array(model.prior.cov)

    def predict(self, current_state_mean, current_state_cov):
        predictive_mean = self.transition_weight @ current_state_mean + self.transition_bias
        predictive_cov = self.transition_weight @ current_state_cov @ self.transition_weight.T + self.transition_cov
        return predictive_mean, predictive_cov
        
    def update(self, predictive_mean, predictive_cov, observation):
        predicted_observation_mean = self.observation_weight @ predictive_mean + self.observation_bias
        predicted_observation_cov = self.observation_weight @ predictive_cov @ self.observation_weight.T + self.observation_cov
        kalman_gain = predictive_cov @ self.observation_weight.T @ np.linalg.inv(predicted_observation_cov)

        corrected_state_mean = predictive_mean + kalman_gain @ (observation - predicted_observation_mean)
        corrected_state_cov = predictive_cov - kalman_gain @ self.observation_weight @ predictive_cov

        return corrected_state_mean, corrected_state_cov

    def filter_step(self, current_state_mean, current_state_cov, observation):
        predicted_mean, predicted_cov = self.predict(current_state_mean, current_state_cov)
        filtered_mean, filtered_cov = self.update(predicted_mean, predicted_cov, observation)
        return predicted_mean, predicted_cov, filtered_mean, filtered_cov
    
    def init(self, observation):
        init_filtering_mean, init_filtering_cov = self.update(self.prior_mean, self.prior_cov, observation)
        return self.prior_mean, self.prior_cov, init_filtering_mean, init_filtering_cov

    def loglikelihood_term(self, predictive_mean, predictive_cov, observation):
        return multivariate_normal(
            mean=self.observation_weight @ predictive_mean + self.observation_bias, 
            cov=self.observation_weight @ predictive_cov @ self.observation_weight.T + self.observation_cov).logpdf(observation)

    def filter(self, observations):
        num_samples = len(observations)
        loglikelihood = 0

        filtering_means = np.zeros((num_samples, self.dim_state))
        filtering_covs = np.zeros((num_samples, self.dim_state, self.dim_state))

        predictive_mean, predictive_cov, filtering_means[0], filtering_covs[0] = self.init(observations[0])
        loglikelihood += self.loglikelihood_term(predictive_mean, predictive_cov, observations[0])

        for sample_nb in range(1, num_samples):
            predictive_mean, predictive_cov, filtering_means[sample_nb], filtering_covs[sample_nb] = self.filter_step(
                                                                            filtering_means[sample_nb-1],
                                                                            filtering_covs[sample_nb-1],
                                                                            observations[sample_nb])

            loglikelihood += self.loglikelihood_term(predictive_mean, predictive_cov, observations[sample_nb])


        return filtering_means, filtering_covs, loglikelihood

