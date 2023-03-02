import jax
from jax import vmap, lax, numpy as jnp
from jax.flatten_util import ravel_pytree as ravel
from .stats.hmm import *
from .utils import *
from backward_ica.stats import BackwardSmoother


class OnlineVariationalAdditiveSmoothing:

    def __init__(self, 
                p:HMM, 
                q:BackwardSmoother, 
                init_carry_func,
                init_func, 
                update_func, 
                additive_functional, 
                normalizer=None, 
                num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples

        if normalizer is None: 
            self.normalizer = lambda x: jnp.exp(x) / num_samples
        else: 
            self.normalizer = normalizer

        self.additive_functional:AdditiveFunctional = additive_functional

        self._init_func = partial(init_func, 
                                p=p, 
                                q=q, 
                                h_0=self.additive_functional.init, 
                                num_samples=num_samples)

        self._update_func = partial(update_func, 
                                    p=p, 
                                    q=q, 
                                    normalizer=self.normalizer, 
                                    h=self.additive_functional.update, 
                                    num_samples=num_samples)

        self._init_carry = partial(init_carry_func, 
                                num_samples=self.num_samples, 
                                out_shape=self.additive_functional.out_shape,
                                state_dim=self.p.state_dim,
                                obs_dim=self.p.obs_dim,
                                dummy_state=self.q.empty_state())
   
    def init_carry(self, params):

        return self._init_carry(params)

    def _init(self, carry, input):
        return self._init_func(carry, input)
        
    def _update(self, carry, input):
        return self._update_func(carry, input)
    
    def compute(self, carry, input):

        carry, output = lax.cond(input['t'] != 0, 
                        self._update, 
                        self._init,
                        carry, input)
        
        return carry, output
    
    def batch_compute(self, key, obs_seq, theta, phi):

        theta.compute_covs()

        T = len(obs_seq) - 1 # T + 1 observations
        keys = jax.random.split(key, T+1) # T+1 keys 
        timesteps = jnp.arange(0, T+1) # [0:T]

        def _step(carry, x):
            t, key_t, obs_t = x
            input_t = {'t':t, 'key':key_t, 'ys':obs_seq, 'y': obs_t, 'phi':phi}            
            carry['theta'] = theta
            carry_t, output_t = self.compute(carry, input_t)
            return carry_t, output_t
        

        carry_m1 = self.init_carry(phi)

        # obs_seq_with_dummy_obs = tree_append(obs_seq, obs_seq[-1])
        # obs_seq_strides = tree_get_strides(2, obs_seq)
        carry, outputs = lax.scan(_step, 
                        init=carry_m1,
                        xs=(timesteps, keys, obs_seq))

        stats = carry['stats']
        # choices = carry['choices']

        # def mean_on_choices(array, choices):
        #     zeros_array = jnp.zeros_like(array)
        #     return jnp.sum(jnp.where(choices, array, zeros_array), axis=0) / choices.sum()

        # return tree_map(lambda x:  jnp.mean(jnp.exp(carry['log_K']) * x, axis=0) / (T+1), stats)['tau'], outputs
        return tree_map(lambda x:  jnp.mean(x, axis=0) / (T+1), stats)['tau'], outputs

        # return tree_map(lambda x: mean_on_choices(x, choices) / (T + 1), stats), outputs
        # weights = self.normalizer(carry['log_q_x'] - carry['log_nu_x'])
        # return tree_map(lambda x: (weights.reshape(-1,1).T @ x).squeeze() / (T + 1), stats), outputs

def init_carry(unformatted_params, state_dim, obs_dim, num_samples, out_shape, dummy_state):


    dummy_tau = jnp.empty((num_samples, *out_shape))
    dummy_x = jnp.empty((num_samples, state_dim)) 
    dummy_log_q_x = jnp.empty((num_samples,))


    return {'state':dummy_state, 
            'log_q_x':dummy_log_q_x, 
            'x':dummy_x, 
            'stats':{'tau':dummy_tau}}

def init_IS(carry_m1, input_0, p:HMM, q:BackwardSmoother, h_0, num_samples):


    y_0 = input_0['y']
    key_0, unformatted_phi_0 = input_0['key'], input_0['phi']

    phi_0 = q.format_params(unformatted_phi_0)

    state_0 = q.init_state(y_0, phi_0)

    filt_params = q.filt_params_from_state(state_0, phi_0)
    x_0, log_q_x_0 = samples_and_log_probs(q.filt_dist, 
                                            key_0, 
                                            filt_params, 
                                            num_samples)

    data_0 = {'x':x_0,
            'log_q_x':log_q_x_0,
            'state':state_0,
            'y':y_0}

    data = {'tm1': carry_m1, 't':data_0}

    tau_0 = named_vmap(partial(h_0, models={'p':p, 'q':q}), 
                    axes_names={'t':{'x':0,'log_q_x':0}}, 
                    input_dict=data)

    carry = {'x':x_0,
            'log_q_x':log_q_x_0,
            'state':state_0,
            'stats':{'tau': tau_0}}

    
    return carry, jnp.zeros((num_samples, num_samples))

def update_IS(
        carry_tm1, 
        input_t:HMM, 
        p:HMM, 
        q:BackwardSmoother, 
        h, 
        num_samples, 
        normalizer):


    state_tm1 = carry_tm1['state']


    t, key_t, y_t, unformatted_phi_t = input_t['t'], input_t['key'], input_t['y'], input_t['phi']

    phi_t = q.format_params(unformatted_phi_t)
    params_q_tm1_t = q.backwd_params_from_state(state_tm1, phi_t)

    state_t = q.new_state(y_t, state_tm1, phi_t)

    tau_tm1 = carry_tm1['stats']['tau']

    h = partial(h, models={'p':p, 'q':q})

    def update(x_t, log_q_t_x_t):

        data_t = {'x':x_t,'log_q_x':log_q_t_x_t, 'y':y_t}

        def weights(x_tm1, log_q_tm1_x_tm1):
            return q.backwd_kernel.logpdf(x_tm1, x_t, params_q_tm1_t) - log_q_tm1_x_tm1
        
        def _h(x_tm1, log_q_tm1_x_tm1):
            data_tm1 = {'x':x_tm1, 'log_q_x':log_q_tm1_x_tm1, 'params_backwd': params_q_tm1_t, 'theta':carry_tm1['theta']}
            data = {'t':data_t, 'tm1':data_tm1}
            return h(data)
        
        w_tm1_t = jax.vmap(weights)(carry_tm1['x'], carry_tm1['log_q_x'])

        h_tm1_t = jax.vmap(_h)(carry_tm1['x'], carry_tm1['log_q_x'])

        return (normalizer(w_tm1_t).reshape(-1,1).T @ (tau_tm1 + h_tm1_t).reshape(-1,1)).squeeze(), w_tm1_t
    
    filt_params = q.filt_params_from_state(state_t, phi_t)
    x_t, log_q_t_x_t = samples_and_log_probs(q.filt_dist, 
                                            key_t, 
                                            filt_params,
                                            num_samples)

            
    tau_t, w_tm1_t = jax.vmap(update)(x_t, log_q_t_x_t)

    carry_t = {'state':state_t, 
            'x':x_t, 
            'stats': {'tau':tau_t},
            'log_q_x':log_q_t_x_t}

    return carry_t, w_tm1_t

def update_PaRIS(
        carry_tm1, 
        input_t:HMM, 
        p:HMM, 
        q:LinearGaussianHMM, 
        h, 
        num_samples, 
        normalizer):


    state_tm1 = carry_tm1['state']


    t, key_t, y_t, unformatted_phi_t = input_t['t'], input_t['key'], input_t['y'], input_t['phi']

    phi_t = q.format_params(unformatted_phi_t)
    params_q_tm1_t = q.backwd_params_from_state(state_tm1, phi_t)

    state_t = q.new_state(y_t, state_tm1, phi_t)

    tau_tm1 = carry_tm1['stats']['tau']

    h = partial(h, models={'p':p, 'q':q})

    x_tm1 = carry_tm1['x']
    log_q_tm1_x_tm1 = carry_tm1['log_q_x']

    def update(x_t, log_q_t_x_t):

        data_t = {'x':x_t,'log_q_x':log_q_t_x_t, 'y':y_t}

        def weights(x_tm1):
            return q.transition_kernel.logpdf(x_t, x_tm1, phi_t.transition)
        
        def _h(x_tm1, log_q_tm1_x_tm1):
            data_tm1 = {'x':x_tm1, 'log_q_x':log_q_tm1_x_tm1, 'params_backwd': params_q_tm1_t, 'theta':carry_tm1['theta']}
            data = {'t':data_t, 'tm1':data_tm1}
            return h(data)
        
        w_tm1_t = jax.vmap(weights)(x_tm1)

        h_tm1_t = jax.vmap(_h)(x_tm1, log_q_tm1_x_tm1)

        return (normalizer(w_tm1_t).reshape(-1,1).T @ (tau_tm1 + h_tm1_t).reshape(-1,1)).squeeze(), w_tm1_t
    
    filt_params = q.filt_params_from_state(state_t, phi_t)
    x_t, log_q_t_x_t = samples_and_log_probs(q.filt_dist, 
                                            key_t, 
                                            filt_params,
                                            num_samples)

            
    tau_t, w_tm1_t = jax.vmap(update)(x_t, log_q_t_x_t)

    carry_t = {'state':state_t, 
            'x':x_t, 
            'stats': {'tau':tau_t},
            'log_q_x':log_q_t_x_t}

    return carry_t, w_tm1_t




OnlinePaRISELBO = lambda p, q, num_samples: OnlineVariationalAdditiveSmoothing(          
                                                    p, 
                                                    q,
                                                    init_carry,
                                                    init_IS,
                                                    update_PaRIS,
                                                    online_elbo_functional(p,q),
                                                    exp_and_normalize,
                                                    num_samples)

OnlineNormalizedISELBO = lambda p, q, num_samples: OnlineVariationalAdditiveSmoothing(          
                                                    p, 
                                                    q,
                                                    init_carry,
                                                    init_IS,
                                                    update_IS,
                                                    online_elbo_functional(p,q),
                                                    exp_and_normalize,
                                                    num_samples)
