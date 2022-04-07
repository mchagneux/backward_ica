from jax import numpy as jnp, lax, config
from jax.scipy.stats.multivariate_normal import logpdf as jax_gaussian_logpdf
from pykalman.standard import KalmanFilter

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)


def kalman_init(obs, prior_params, emission_params):
    filt_mean, filt_cov = kalman_update(prior_params.mean, prior_params.cov, obs, emission_params)
    return filt_mean, filt_cov

def kalman_predict(filt_mean, filt_cov, transition_params):
    A, a, Q = transition_params.matrix, transition_params.bias, transition_params.cov
    pred_mean = A @ filt_mean + a
    pred_cov = A @ filt_cov @ A.T + Q
    return pred_mean, pred_cov

def kalman_update(pred_mean, pred_cov, obs, emission_params):

    B, b, R = emission_params.matrix, emission_params.bias, emission_params.cov 
    kalman_gain = pred_cov @ B.T @ jnp.linalg.inv(B @ pred_cov @ B.T + R)

    filt_mean = pred_mean + kalman_gain @ (obs - (B @ pred_mean + b))
    filt_cov = pred_cov - kalman_gain @ B @ pred_cov

    return filt_mean, filt_cov

def kalman_filter_seq(obs_seq, hmm_params):

    def log_l_term(pred_mean, pred_cov, obs, emission_params):
        B, b, R = emission_params.matrix, emission_params.bias, emission_params.cov
        return jax_gaussian_logpdf(x=obs, 
                            mean=B @ pred_mean + b , 
                            cov=B @ pred_cov @ B.T + R)

    init_filt_mean, init_filt_cov = kalman_init(obs_seq[0], hmm_params.prior, hmm_params.emission)
    loglikelihood = log_l_term(hmm_params.prior.mean, hmm_params.prior.cov, obs_seq[0], hmm_params.emission)

    def _filter_step(carry, x):
        loglikelihood, filt_mean, filt_cov, transition_params, emission_params  = carry
        pred_mean, pred_cov = kalman_predict(filt_mean, filt_cov, transition_params)
        filt_mean, filt_cov = kalman_update(pred_mean, pred_cov, x, emission_params)

        loglikelihood += log_l_term(pred_mean, pred_cov, x, emission_params)

        return (loglikelihood, filt_mean, filt_cov, transition_params, emission_params), (pred_mean, pred_cov, filt_mean, filt_cov)

    (loglikelihood, *_), (pred_mean_seq, pred_cov_seq, filt_mean_seq, filt_cov_seq) = lax.scan(f=_filter_step, 
                                init=(loglikelihood, init_filt_mean, init_filt_cov, hmm_params.transition, hmm_params.emission), 
                                xs=obs_seq[1:])

    pred_mean_seq = jnp.concatenate((hmm_params.prior.mean[None,:], pred_mean_seq))
    pred_cov_seq = jnp.concatenate((hmm_params.prior.cov[None,:], pred_cov_seq))
    filt_mean_seq =  jnp.concatenate((init_filt_mean[None,:], filt_mean_seq))
    filt_cov_seq =  jnp.concatenate((init_filt_cov[None,:], filt_cov_seq))

    return pred_mean_seq, pred_cov_seq, filt_mean_seq, filt_cov_seq, loglikelihood

def kalman_smooth_seq(obs_seq, hmm_params):

    pred_mean_seq, pred_cov_seq, filt_mean_seq, filt_cov_seq = kalman_filter_seq(obs_seq, hmm_params)[:-1]

    last_smooth_mean, last_smooth_cov = filt_mean_seq[-1], filt_cov_seq[-1]

    def _smooth_step(carry, x):
        smooth_mean, smooth_cov, transition_matrix = carry 
        filt_mean, filt_cov, next_pred_mean, next_pred_cov = x  
        
        C = filt_cov @ transition_matrix @ jnp.linalg.inv(next_pred_cov)
        smooth_mean = filt_mean + C @ (smooth_mean - next_pred_mean)
        smooth_cov = filt_cov + C @ (smooth_cov - next_pred_cov) @ C.T

        return (smooth_mean, smooth_cov, transition_matrix), (smooth_mean, smooth_cov)

    _, (smooth_mean_seq, smooth_cov_seq) = lax.scan(f=_smooth_step,
                                            init=(last_smooth_mean, last_smooth_cov, hmm_params.transition.matrix),
                                            xs=(filt_mean_seq[:-1], 
                                                filt_cov_seq[:-1],
                                                pred_mean_seq[1:],
                                                pred_cov_seq[1:]),
                                            reverse=True)

    smooth_mean_seq = jnp.concatenate((smooth_mean_seq, last_smooth_mean[None,:]))
    smooth_cov_seq = jnp.concatenate((smooth_cov_seq, last_smooth_cov[None,:]))

    return smooth_mean_seq, smooth_cov_seq



def pykalman_filter_seq(obs_seq, hmm_params):

    engine = KalmanFilter(transition_matrices=hmm_params.transition.matrix, 
                        observation_matrices=hmm_params.emission.matrix,
                        transition_covariance=hmm_params.transition.cov,
                        observation_covariance=hmm_params.emission.cov,
                        transition_offsets=hmm_params.transition.bias,
                        observation_offsets=hmm_params.emission.bias,
                        initial_state_mean=hmm_params.prior.mean,
                        initial_state_covariance=hmm_params.prior.cov)

    return engine.filter(obs_seq)

def pykalman_smooth_seq(obs_seq, hmm_params):
    engine = KalmanFilter(transition_matrices=hmm_params.transition.matrix, 
                        observation_matrices=hmm_params.emission.matrix,
                        transition_covariance=hmm_params.transition.cov,
                        observation_covariance=hmm_params.emission.cov,
                        transition_offsets=hmm_params.transition.bias,
                        observation_offsets=hmm_params.emission.bias,
                        initial_state_mean=hmm_params.prior.mean,
                        initial_state_covariance=hmm_params.prior.cov)
    return engine.smooth(obs_seq)