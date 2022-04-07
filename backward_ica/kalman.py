from jax import numpy as jnp, lax, config
from jax.scipy.stats.multivariate_normal import logpdf as jax_gaussian_logpdf
from pykalman.standard import KalmanFilter

from backward_ica.hmm import GaussianHMM

from .utils import * 

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

def predict(filt_mean, filt_cov, transition:LinearGaussianKernel):
    pred_mean = transition.map(filt_mean)
    pred_cov = transition.weight @ filt_cov @ transition.weight.T + transition.cov
    return pred_mean, pred_cov

def update(pred_mean, pred_cov, obs, emission:LinearGaussianKernel):

    kalman_gain = pred_cov @ emission.weight.T @ jnp.linalg.inv(emission.weight @ pred_cov @ emission.weight.T + emission.cov)

    filt_mean = pred_mean + kalman_gain @ (obs - emission.map(pred_mean))
    filt_cov = pred_cov - kalman_gain @ emission.weight @ pred_cov

    return filt_mean, filt_cov

def filter_step(filt_mean, filt_cov, obs, transition:LinearGaussianKernel, emission:LinearGaussianKernel):
    pred_mean, pred_cov = predict(filt_mean, filt_cov, transition)
    filt_mean, filt_cov = update(pred_mean, pred_cov, obs, emission)
    return pred_mean, pred_cov, filt_mean, filt_cov

def init(obs, prior:Gaussian, emission:LinearGaussianKernel):
    filt_mean, filt_cov = update(prior.mean, prior.cov, obs, emission)
    return prior.mean, prior.cov, filt_mean, filt_cov

def log_l_term(pred_mean, pred_cov, obs, emission:LinearGaussianKernel):
    return jax_gaussian_logpdf(x=obs, 
                        mean=emission.map(pred_mean), 
                        cov=emission.weight @ pred_cov @ emission.weight.T + emission.cov)


def filter(obs_seq, hmm:GaussianHMM):
    init_pred_mean, init_pred_cov, init_filt_mean, init_filt_cov = init(obs_seq[0], hmm.prior, hmm.emission)
    loglikelihood = log_l_term(init_pred_mean, init_pred_cov, obs_seq[0], hmm.emission)

    def _filter_step(carry, x):
        loglikelihood, filt_mean, filt_cov, transition, emission  = carry
        pred_mean, pred_cov, filt_mean, filt_cov = filter_step(filt_mean=filt_mean,
                                                                        filt_cov=filt_cov,
                                                                        obs=x,
                                                                        transition=transition,
                                                                        emission=emission)

        loglikelihood += log_l_term(pred_mean, pred_cov, x, emission)

        return (loglikelihood, filt_mean, filt_cov, transition, emission), (pred_mean, pred_cov, filt_mean, filt_cov)

    (loglikelihood, *_), (pred_mean_seq, pred_cov_seq, filt_mean_seq, filt_cov_seq) = lax.scan(f=_filter_step, 
                                init=(loglikelihood, init_filt_mean, init_filt_cov, hmm.transition, hmm.emission), 
                                xs=obs_seq[1:])

    pred_mean_seq = jnp.concatenate((init_pred_mean[None,:], pred_mean_seq))
    pred_cov_seq = jnp.concatenate((init_pred_cov[None,:], pred_cov_seq))
    filt_mean_seq =  jnp.concatenate((init_filt_mean[None,:], filt_mean_seq))
    filt_cov_seq =  jnp.concatenate((init_filt_cov[None,:], filt_cov_seq))

    return pred_mean_seq, pred_cov_seq, filt_mean_seq, filt_cov_seq, loglikelihood



def smooth(obs_seq, hmm:GaussianHMM):

    pred_mean_seq, pred_cov_seq, filt_mean_seq, filt_cov_seq = filter(obs_seq, hmm)[:-1]

    last_smooth_mean, last_smooth_cov = filt_mean_seq[-1], filt_cov_seq[-1]

    def _smooth_step(carry, x):
        smooth_mean, smooth_cov, transition_matrix = carry 
        filt_mean, filt_cov, next_pred_mean, next_pred_cov = x  
        
        C = filt_cov @ transition_matrix @ jnp.linalg.inv(next_pred_cov)
        smooth_mean = filt_mean + C @ (smooth_mean - next_pred_mean)
        smooth_cov = filt_cov + C @ (smooth_cov - next_pred_cov) @ C.T

        return (smooth_mean, smooth_cov, transition_matrix), (smooth_mean, smooth_cov)

    _, (smooth_mean_seq, smooth_cov_seq) = lax.scan(f=_smooth_step,
                                            init=(last_smooth_mean, last_smooth_cov, hmm.transition.weight),
                                            xs=(filt_mean_seq[:-1], 
                                                filt_cov_seq[:-1],
                                                pred_mean_seq[1:],
                                                pred_cov_seq[1:]),
                                            reverse=True)

    smooth_mean_seq = jnp.concatenate((smooth_mean_seq, last_smooth_mean[None,:]))
    smooth_cov_seq = jnp.concatenate((smooth_cov_seq, last_smooth_cov[None,:]))

    return smooth_mean_seq, smooth_cov_seq



def filter_pykalman(obs_seq, hmm:GaussianHMM):

    engine = KalmanFilter(transition_matrices=hmm.transition.weight, 
                        observation_matrices=hmm.emission.weight,
                        transition_covariance=hmm.transition.cov,
                        observation_covariance=hmm.emission.cov,
                        transition_offsets=hmm.transition.bias,
                        observation_offsets=hmm.emission.bias,
                        initial_state_mean=hmm.prior.mean,
                        initial_state_covariance=hmm.prior.cov)

    return engine.filter(obs_seq)

def smooth_pykalman(obs_seq, hmm:GaussianHMM):
    engine = KalmanFilter(transition_matrices=hmm.transition.weight, 
                        observation_matrices=hmm.emission.weight,
                        transition_covariance=hmm.transition.cov,
                        observation_covariance=hmm.emission.cov,
                        transition_offsets=hmm.transition.bias,
                        observation_offsets=hmm.emission.bias,
                        initial_state_mean=hmm.prior.mean,
                        initial_state_covariance=hmm.prior.cov)
    return engine.smooth(obs_seq)