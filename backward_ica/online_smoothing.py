from abc import ABCMeta, abstractmethod
import jax
from jax import vmap, lax, numpy as jnp
from .stats.hmm import *
from .utils import *
from backward_ica.stats import BackwardSmoother


class OnlineVariationalAdditiveSmoothing(metaclass=ABCMeta):

    def __init__(self, p:HMM, q:BackwardSmoother, functionals, normalizer=None, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples

        if normalizer is None: 
            self.normalizer = lambda x: jnp.exp(x) / self.num_samples
        else: 
            self.normalizer = normalizer

        self.h_0, self.h_t, self.stat_shape = functionals(p, q)

        self.init_carry = self.init()

    def init(self):

        dummy_tau = jnp.empty((self.num_samples, *self.stat_shape))
        dummy_x = jnp.empty((self.num_samples, self.p.state_dim)) 
        dummy_log_q = jnp.empty((self.num_samples,))
        dummy_state = self.q.empty_state()

        return dummy_state, dummy_tau, dummy_x, dummy_log_q


    def samples_and_log_probs(self, key, q_params):
        samples = vmap(partial(self.q.filt_dist.sample, params=q_params))(random.split(key, self.num_samples))
        log_probs = vmap(partial(self.q.filt_dist.logpdf, params=q_params))(samples)
        return samples, log_probs

    def _init(self, key_0, dummy_state, dummy_tau, dummy_x, dummy_log_q, y_0, theta, phi):

        state_0 = self.q.init_state(y_0, phi)
        x_0, log_q_x_0 = self.samples_and_log_probs(key_0, 
                                                    self.q.filt_params_from_state(state_0, phi))

        tau_0 = vmap(partial(self.h_0, y_0=y_0, theta=theta, log_q_x_0=log_q_x_0))(x_0)

        return state_0, tau_0, x_0, log_q_x_0

    @abstractmethod
    def _update(self, key_t, state_tm1, tau_tm1, x_tm1, log_q_tm1_x_tm1, y_t, theta, phi):
        raise NotImplementedError
    
    def compute(self, t, key_t, state_tm1, tau_tm1, x_tm1, log_q_tm1_x_tm1, y_t, theta, phi):

        (state_t, tau_t, x_t, log_q_t_x_t) = lax.cond(t != 0, 
                                                    self._update, 
                                                    self._init,
                                                    key_t, state_tm1, tau_tm1, x_tm1, log_q_tm1_x_tm1, y_t, theta, phi)
        
        return (state_t, tau_t, x_t, log_q_t_x_t), None
    
    def batch_compute(self, key, obs_seq, theta, phi):

        keys = jax.random.split(key, len(obs_seq))
        timesteps = jnp.arange(0, len(obs_seq))

        def _step(carry, x):
            state_tm1, tau_tm1, x_tm1, log_q_tm1_x_tm1 = carry
            t, key_t, y_t = x
            return self.compute(t, key_t, state_tm1, tau_tm1, x_tm1, log_q_tm1_x_tm1, y_t, theta, phi)
        
        tau_T = lax.scan(_step, 
                        init=self.init_carry,
                        xs=(timesteps, keys, obs_seq))[0][1] 
                        
        return jnp.mean(tau_T, axis=0) / len(obs_seq)

class OnlineISAdditiveSmoothing(OnlineVariationalAdditiveSmoothing):

    def __init__(self, p:HMM, q:BackwardSmoother, functionals, normalizer=None, num_samples=200):

        super().__init__(p, q, functionals, normalizer, num_samples)


    def _update(self, key_t, state_tm1, tau_tm1, x_tm1, log_q_tm1_x_tm1, y_t, theta, phi):

        q_tm1_t_params = self.q.backwd_params_from_state(state_tm1, phi)

        def compute_tau_t(x_t, log_q_t_x_t):
            def compute_sum_term(x_tm1, log_q_tm1_x_tm1, tau_tm1):

                importance_log_weight = self.q.backwd_kernel.logpdf(x_tm1, x_t, q_tm1_t_params) \
                                        - log_q_tm1_x_tm1

                sum_term = tau_tm1 + self.h_t(x_tm1=x_tm1, 
                                                x_t=x_t, 
                                                y_t=y_t,
                                                log_q_tm1_x_tm1=log_q_tm1_x_tm1,
                                                log_q_t_x_t=log_q_t_x_t,
                                                theta=theta,
                                                phi=phi,
                                                q_tm1_t_params=q_tm1_t_params)

                return importance_log_weight, sum_term

            importance_log_weights, sum_terms = vmap(compute_sum_term)(x_tm1, log_q_tm1_x_tm1, tau_tm1)

            normalized_importance_weights = self.normalizer(importance_log_weights)
            return normalized_importance_weights @ sum_terms
            
        state_t = self.q.new_state(y_t, state_tm1, phi)

        x_t, log_q_t_x_t = self.samples_and_log_probs(key_t, 
                                                    self.q.filt_params_from_state(state_t, phi))


        tau_t = vmap(compute_tau_t)(x_t, log_q_t_x_t) 

        return state_t, tau_t, x_t, log_q_t_x_t
    
class OnlinePaRISAdditiveSmoothing(OnlineVariationalAdditiveSmoothing):

    def __init__(self, p:HMM, q:BackwardSmoother, functionals, normalizer=None, num_samples=200):

        super().__init__(p, q, functionals, normalizer, num_samples)
        self.num_paris_samples = 2 


    def _update(self, key_t, state_tm1, tau_tm1, x_tm1, log_q_tm1_x_tm1, y_t, theta, phi):

        q_tm1_t_params = self.q.backwd_params_from_state(state_tm1, phi)

        def compute_tau_t(key, x_t, log_q_t_x_t):

            def compute_sum_term(x_tm1, log_q_tm1_x_tm1, tau_tm1):
                sum_term = tau_tm1 + self.h_t(x_tm1=x_tm1, 
                                                x_t=x_t, 
                                                y_t=y_t,
                                                log_q_tm1_x_tm1=log_q_tm1_x_tm1,
                                                log_q_t_x_t=log_q_t_x_t,
                                                theta=theta,
                                                phi=phi,
                                                q_tm1_t_params=q_tm1_t_params)
                return sum_term 


            log_q_tm1_t_x_t = vmap(partial(self.q.backwd_kernel.logpdf, 
                                            state=x_t, 
                                            params=q_tm1_t_params))(x_tm1)

            normalized_q_tm1_t_x_t = self.normalizer(log_q_tm1_t_x_t - log_q_tm1_x_tm1)

            ancestor_indices = jax.random.choice(key, 
                                                a=self.num_samples, 
                                                shape=(self.num_paris_samples,), 
                                                p=normalized_q_tm1_t_x_t)

            sum_terms = vmap(compute_sum_term)(x_tm1[ancestor_indices], 
                                            log_q_tm1_x_tm1[ancestor_indices], 
                                            tau_tm1[ancestor_indices])

            return jnp.mean(sum_terms, axis=0)
            
        state_t = self.q.new_state(y_t, state_tm1, phi)

        key_new_samples, key_paris = random.split(key_t, 2)

        x_t, log_q_t_x_t = self.samples_and_log_probs(key_new_samples, 
                                                    self.q.filt_params_from_state(state_t, phi))

        tau_t = vmap(compute_tau_t)(random.split(key_paris, self.num_samples), 
                                    x_t, 
                                    log_q_t_x_t) 

        return state_t, tau_t, x_t, log_q_t_x_t
