from jax import numpy as jnp, lax, jit
from jax.scipy.stats.multivariate_normal import logpdf as jax_gaussian_logpdf
from pykalman.standard import KalmanFilter
from backward_ica.utils import * 


class Kalman: 


    def init(obs, prior_params, emission_params):
        filt_mean, filt_cov = Kalman.update(prior_params.mean, prior_params.scale.cov, obs, emission_params)
        return filt_mean, filt_cov

    def predict(filt_mean, filt_cov, transition_params):
        A, a, Q = transition_params.map.w, transition_params.map.b, transition_params.noise.scale.cov
        pred_mean = A @ filt_mean + a
        pred_cov = A @ filt_cov @ A.T + Q
        return pred_mean, pred_cov

    def update(pred_mean, pred_cov, obs, emission_params):

        B, b, R = emission_params.map.w, emission_params.map.b, emission_params.noise.scale.cov 
        kalman_gain = pred_cov @ B.T @ inv(B @ pred_cov @ B.T + R)

        filt_mean = pred_mean + kalman_gain @ (obs - (B @ pred_mean + b))
        filt_cov = pred_cov - kalman_gain @ B @ pred_cov

        return filt_mean, filt_cov

    def filter_seq(obs_seq, hmm_params):
        
        def pred_log_l_term(pred_mean, pred_cov, obs, emission_params):
            B, b, R = emission_params.map.w, emission_params.map.b, emission_params.noise.scale.cov
            return jax_gaussian_logpdf(x=obs, 
                                mean=B @ pred_mean + b , 
                                cov=B @ pred_cov @ B.T + R)

        init_filt_mean, init_filt_cov = Kalman.init(obs_seq[0], hmm_params.prior, hmm_params.emission)
        init_pred_loglikelihood = pred_log_l_term(hmm_params.prior.mean, hmm_params.prior.scale.cov, obs_seq[0], hmm_params.emission)

        @jit
        def _filter_step(carry, x):
            filt_mean, filt_cov = carry
            pred_mean, pred_cov = Kalman.predict(filt_mean, filt_cov, hmm_params.transition)
            filt_mean, filt_cov = Kalman.update(pred_mean, pred_cov, x, hmm_params.emission)

            pred_loglikelihood = pred_log_l_term(pred_mean, pred_cov, x, hmm_params.emission)

            return (filt_mean, filt_cov), (pred_mean, pred_cov, filt_mean, filt_cov, pred_loglikelihood)

        (pred_mean_seq, pred_cov_seq, filt_mean_seq, filt_cov_seq, pred_loglikelihood_seq) = lax.scan(f=_filter_step, 
                                    init=(init_filt_mean, init_filt_cov), 
                                    xs=obs_seq[1:])[1]

        pred_mean_seq = tree_prepend(hmm_params.prior.mean, pred_mean_seq) 
        pred_cov_seq =  tree_prepend(hmm_params.prior.scale.cov, pred_cov_seq) 
        filt_mean_seq = tree_prepend(init_filt_mean, filt_mean_seq) 
        filt_cov_seq =  tree_prepend(init_filt_cov, filt_cov_seq)

        return pred_mean_seq, pred_cov_seq, filt_mean_seq, filt_cov_seq, init_pred_loglikelihood + jnp.sum(pred_loglikelihood_seq)

    def smooth_seq(obs_seq, hmm_params):

        pred_mean_seq, pred_cov_seq, filt_mean_seq, filt_cov_seq = Kalman.filter_seq(obs_seq, hmm_params)[:-1]

        last_smooth_mean, last_smooth_cov = filt_mean_seq[-1], filt_cov_seq[-1]
        
        @jit
        def _smooth_step(carry, x):
            next_smooth_mean, next_smooth_cov = carry 
            filt_mean, filt_cov, next_pred_mean, next_pred_cov = x  
            
            C = filt_cov @ hmm_params.transition.map.w @ inv(next_pred_cov)
            smooth_mean = filt_mean + C @ (next_smooth_mean - next_pred_mean)
            smooth_cov = filt_cov + C @ (next_smooth_cov - next_pred_cov) @ C.T

            return (smooth_mean, smooth_cov), (smooth_mean, smooth_cov)

        _, (smooth_mean_seq, smooth_cov_seq) = lax.scan(f=_smooth_step,
                                                init=(last_smooth_mean, last_smooth_cov),
                                                xs=(filt_mean_seq[:-1], 
                                                    filt_cov_seq[:-1],
                                                    pred_mean_seq[1:],
                                                    pred_cov_seq[1:]),
                                                reverse=True)

        smooth_mean_seq = jnp.concatenate((smooth_mean_seq, last_smooth_mean[None,:]))
        smooth_cov_seq = jnp.concatenate((smooth_cov_seq, last_smooth_cov[None,:]))

        return smooth_mean_seq, smooth_cov_seq






def pykalman_filter_seq(obs_seq, hmm_params):

    engine = KalmanFilter(transition_matrices=hmm_params.transition.map.w, 
                        observation_matrices=hmm_params.emission.map.w,
                        transition_covariance=hmm_params.transition.noise.scale.cov,
                        observation_covariance=hmm_params.emission.noise.scale.cov,
                        transition_offsets=hmm_params.transition.map.b,
                        observation_offsets=hmm_params.emission.map.b,
                        initial_state_mean=hmm_params.prior.mean,
                        initial_state_covariance=hmm_params.prior.scale.cov)

    return engine.filter(obs_seq)

def pykalman_logl_seq(obs_seq, hmm_params):


    engine = KalmanFilter(transition_matrices=jnp.asarray(hmm_params.transition.map.w), 
                        observation_matrices=jnp.asarray(hmm_params.emission.map.w),
                        transition_covariance=jnp.asarray(hmm_params.transition.noise.scale.cov),
                        observation_covariance=jnp.asarray(hmm_params.emission.noise.scale.cov),
                        transition_offsets=jnp.asarray(hmm_params.transition.map.b),
                        observation_offsets=jnp.asarray(hmm_params.emission.map.b),
                        initial_state_mean=jnp.asarray(hmm_params.prior.mean),
                        initial_state_covariance=jnp.asarray(hmm_params.prior.scale.cov))

    
    return engine.loglikelihood(obs_seq)
    
def pykalman_smooth_seq(obs_seq, hmm_params):

    engine = KalmanFilter(transition_matrices=hmm_params.transition.map.w, 
                        observation_matrices=hmm_params.emission.map.w,
                        transition_covariance=hmm_params.transition.noise.scale.cov,
                        observation_covariance=hmm_params.emission.noise.scale.cov,
                        transition_offsets=hmm_params.transition.map.b,
                        observation_offsets=hmm_params.emission.map.b,
                        initial_state_mean=hmm_params.prior.mean,
                        initial_state_covariance=hmm_params.prior.scale.cov)
                        
    return engine.smooth(obs_seq)