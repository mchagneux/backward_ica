import jax
from jax import vmap, lax, numpy as jnp
from .stats.hmm import *
from .utils import *
from backward_ica.stats import BackwardSmoother


class OnlineVariationalAdditiveSmoothing:

    def __init__(self, 
                p:HMM, 
                q:BackwardSmoother, 
                init_func, 
                update_func, 
                additive_functional, 
                normalizer=None, 
                num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples

        if normalizer is None: 
            self.normalizer = lambda x: jnp.exp(x) / self.num_samples
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

    def init_carry(self):

        dummy_tau = jnp.empty((self.num_samples, *self.additive_functional.out_shape))
        dummy_x = jnp.empty((self.num_samples, self.p.state_dim)) 
        dummy_y = jnp.empty((self.p.obs_dim,))
        dummy_log_q_x = jnp.empty((self.num_samples,))
        dummy_state = self.q.empty_state()

        return {'state':dummy_state, 
                'y':dummy_y,
                'log_q_x':dummy_log_q_x, 
                'x':dummy_x, 
                'tau':dummy_tau}

    def _init(self, carry, input):
        return self._init_func(carry, input)
        
    def _update(self, carry, input):
        return self._update_func(carry, input)
    
    def compute(self, carry, input):

        carry = lax.cond(input['t'] != 0, 
                        self._update, 
                        self._init,
                        carry, input)
        
        return carry, None
    
    def batch_compute(self, key, obs_seq, theta, phi):

        theta.compute_covs()
        phi.compute_covs()

        T = len(obs_seq) - 1 # T + 1 observations
        keys = jax.random.split(key, T+1) # T+1 keys 
        timesteps = jnp.arange(0, T+1) # [0:T]

        def _step(carry, x):
            t, key_t, y_t = x
            input_t = {'t':t, 'key':key_t, 'y':y_t}
            carry_t, output_t = self.compute(carry, input_t)
            carry_t['theta'] = theta
            carry_t['phi'] = phi
            return carry_t, output_t
        

        carry_m1 = self.init_carry()
        carry_m1['theta'] = theta
        carry_m1['phi'] = phi

        tau_T = lax.scan(_step, 
                        init=carry_m1,
                        xs=(timesteps, keys, obs_seq))[0]['tau']
                        
        return jnp.mean(tau_T, axis=0) / (T+1)




def init_standard(carry_m1, input_0, p:HMM, q:BackwardSmoother, h_0, num_samples):


    y_0 = input_0['y']
    key_0 = input_0['key']

    phi = carry_m1['phi']

    state_0 = q.init_state(y_0, carry_m1['phi'])

    x_0, log_q_x_0 = samples_and_log_probs(q.filt_dist, 
                                            key_0, 
                                            q.filt_params_from_state(state_0, phi), 
                                            num_samples)

    data_0 = {'x':x_0,
            'log_q_x':log_q_x_0,
            'state':state_0,
            'y':y_0}

    data = {'tm1': carry_m1, 't':data_0}

    tau_0 = named_vmap(partial(h_0, models={'p':p, 'q':q}), 
                    axes_names={'t':{'x':0}}, 
                    input_dict=data)

    data_0['tau'] = tau_0
    
    return data_0

def update_IS(carry_tm1, input_t:HMM, p:HMM, q:BackwardSmoother, h, num_samples, normalizer):


    state_tm1 = carry_tm1['state']
    phi = carry_tm1['phi']

    params_q_tm1_t = q.backwd_params_from_state(state_tm1, phi)

    key_t, y_t = input_t['key'], input_t['y']

    state_t = q.new_state(y_t, state_tm1, phi)

    data_tm1 = {'x':carry_tm1['x'],
                'log_q_x':carry_tm1['log_q_x'],
                'tau':carry_tm1['tau'],
                'theta':carry_tm1['theta'],
                'params_backwd':params_q_tm1_t}

    h = partial(h, models={'p':p, 'q':q})

    def compute_tau_t(data_t):

        def compute_sum_term(data_tm1):

            importance_log_weight = q.backwd_kernel.logpdf(data_tm1['x'], data_t['x'], params_q_tm1_t) \
                                    - data_tm1['log_q_x']

            data = {'tm1':data_tm1, 
                    't': data_t}

            sum_term = data_tm1['tau'] + h(data)

            return importance_log_weight, sum_term

        importance_log_weights, sum_terms = named_vmap(compute_sum_term, 
                                                        axes_names={'x':0, 'log_q_x':0, 'tau':0}, 
                                                        input_dict=data_tm1)

        normalized_importance_weights = normalizer(importance_log_weights)

        return normalized_importance_weights @ sum_terms
        

    
    x_t, log_q_t_x_t = samples_and_log_probs(q.filt_dist, 
                                            key_t, 
                                            q.filt_params_from_state(state_t, phi),
                                            num_samples)

    data_t = {'state':state_t, 
            'x':x_t, 
            'y':y_t,
            'log_q_x':log_q_t_x_t}
            
    tau_t = named_vmap(compute_tau_t, axes_names={'x':0,'log_q_x':0}, input_dict=data_t)

    carry_t = {'state':state_t, 
            'x':x_t, 
            'y':y_t,
            'tau': tau_t,
            'log_q_x':log_q_t_x_t}

    return carry_t
    
def update_PaRIS(carry_tm1, input_t:HMM, p:HMM, q:BackwardSmoother, h, num_samples, normalizer, num_paris_samples):


    state_tm1 = carry_tm1['state']
    phi = carry_tm1['phi']

    params_q_tm1_t = q.backwd_params_from_state(state_tm1, phi)

    key_t, y_t = input_t['key'], input_t['y']

    state_t = q.new_state(y_t, state_tm1, phi)

    h = partial(h, models={'p':p, 'q':q})

    def compute_tau_t(data_t):

        def compute_sum_term(data_tm1_ancestors):

            data = {'tm1':data_tm1_ancestors, 
                    't': data_t}

            sum_term = data_tm1_ancestors['tau'] + h(data)

            return sum_term 


        log_q_tm1_t_x_t = vmap(partial(q.backwd_kernel.logpdf, 
                                        state=data_t['x'], 
                                        params=params_q_tm1_t))(carry_tm1['x'])

        normalized_q_tm1_t_x_t = normalizer(log_q_tm1_t_x_t - carry_tm1['log_q_x'])

        ancestor_indices = jax.random.choice(data_t['key_ancestors'], 
                                            a=num_samples, 
                                            shape=(num_paris_samples,), 
                                            p=normalized_q_tm1_t_x_t)


        data_tm1_ancestors = {'x':carry_tm1['x'][ancestor_indices], 
                            'log_q_x':carry_tm1['log_q_x'][ancestor_indices], 
                            'tau':carry_tm1['tau'][ancestor_indices],
                            'params_backwd':params_q_tm1_t,
                            'theta':carry_tm1['theta']}

        sum_terms = named_vmap(compute_sum_term, 
                            axes_names={'x':0, 'tau':0, 'log_q_x':0}, 
                            input_dict=data_tm1_ancestors)

        return jnp.mean(sum_terms, axis=0)
        
    state_t = q.new_state(y_t, state_tm1, phi)

    key_q_t, *keys_ancestors = jax.random.split(key_t, num_samples + 1)

    x_t, log_q_t_x_t = samples_and_log_probs(q.filt_dist, 
                                            key_q_t, 
                                            q.filt_params_from_state(state_t, phi),
                                            num_samples)

    data_t = {'x': x_t, 
            'log_q_x':log_q_t_x_t, 
            'key_ancestors': jnp.array(keys_ancestors),
            'y':y_t}

    tau_t = named_vmap(compute_tau_t, 
                    axes_names={'x':0, 'log_q_x':0, 'key_ancestors':0}, 
                    input_dict=data_t)

    carry_t = {'x': x_t, 
            'state':state_t,
            'log_q_x':log_q_t_x_t, 
            'tau':tau_t,
            'y':y_t}

    return carry_t
