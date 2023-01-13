import jax
from jax import vmap, lax, numpy as jnp
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

        self._init_carry = partial(init_carry_func, 
                                num_samples=self.num_samples, 
                                out_shape=self.additive_functional.out_shape,
                                state_dim=self.p.state_dim,
                                obs_dim=self.p.obs_dim,
                                dummy_state=self.q.dummy_state())

                    

    def init_carry(self, params):

        return self._init_carry(params)

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
        phi = self.q.format_params(phi)
        phi.compute_covs()

        T = len(obs_seq) - 1 # T + 1 observations
        keys = jax.random.split(key, T+1) # T+1 keys 
        timesteps = jnp.arange(0, T+1) # [0:T]

        def _step(carry, x):
            t, key_t, y_t = x
            input_t = {'t':t, 'key':key_t, 'y':y_t, 'phi':phi}
            carry_t, output_t = self.compute(carry, input_t)
            carry_t['theta'] = theta
            return carry_t, output_t
        

        carry_m1 = self.init_carry(phi)
        carry_m1['theta'] = theta

        tau_T = lax.scan(_step, 
                        init=carry_m1,
                        xs=(timesteps, keys, obs_seq))[0]['tau']
                        
        return jnp.mean(tau_T, axis=0) / (T+1)


def init_carry(unformatted_params, state_dim, obs_dim, num_samples, out_shape, empty_state):


    dummy_tau = jnp.empty((num_samples, out_shape))
    dummy_x = jnp.empty((num_samples, state_dim)) 
    dummy_y = jnp.empty((obs_dim,))
    dummy_log_q_x = jnp.empty((num_samples,))
    dummy_state = empty_state


    return {'state':dummy_state, 
            'y':dummy_y,
            'log_q_x':dummy_log_q_x, 
            'x':dummy_x, 
            'phi':unformatted_params,
            'tau':dummy_tau}


def init_carry_gradients(unformatted_params, state_dim, obs_dim, num_samples, out_shape, empty_state):

    dummy_tau = jnp.empty((num_samples, out_shape))
    dummy_x = jnp.empty((num_samples, state_dim)) 
    dummy_y = jnp.empty((obs_dim,))
    dummy_log_q_x = jnp.empty((num_samples,))
    dummy_state = empty_state
    dummy_grads = jax.vmap(jax.grad(lambda x, params: x), argnums=(1,))(dummy_x, unformatted_params)

    return {'state':dummy_state, 
            'y':dummy_y,
            'log_q_x':dummy_log_q_x, 
            'x':dummy_x, 
            'phi':unformatted_params,
            'tau':dummy_tau,
            'omega':dummy_grads}


def init_standard(carry_m1, input_0, p:HMM, q:BackwardSmoother, h_0, num_samples):


    y_0 = input_0['y']
    key_0, phi_0 = input_0['key'], input_0['phi']

    state_0 = q.init_state(y_0, phi_0)

    x_0, log_q_x_0 = samples_and_log_probs(q.filt_dist, 
                                            key_0, 
                                            q.filt_params_from_state(state_0, phi_0), 
                                            num_samples)

    data_0 = {'x':x_0,
            'log_q_x':log_q_x_0,
            'state':state_0,
            'phi':phi_0,
            'y':y_0}

    data = {'tm1': carry_m1, 't':data_0}

    tau_0 = named_vmap(partial(h_0, models={'p':p, 'q':q}), 
                    axes_names={'t':{'x':0}}, 
                    input_dict=data)

    data_0['tau'] = tau_0
    
    return data_0

def update_IS(
        carry_tm1, 
        input_t:HMM, 
        p:HMM, 
        q:BackwardSmoother, 
        h, 
        num_samples, 
        normalizer):


    state_tm1 = carry_tm1['state']
    phi_tm1 = carry_tm1['phi']

    params_q_tm1_t = q.backwd_params_from_state(state_tm1, phi_tm1)

    key_t, y_t, phi_t = input_t['key'], input_t['y'], input_t['phi']

    state_t = q.new_state(y_t, state_tm1, phi_t)

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
                                            q.filt_params_from_state(state_t, phi_t),
                                            num_samples)

    data_t = {'state':state_t, 
            'x':x_t, 
            'y':y_t,
            'log_q_x':log_q_t_x_t}
            
    tau_t = named_vmap(compute_tau_t, axes_names={'x':0,'log_q_x':0}, input_dict=data_t)

    carry_t = {'state':state_t, 
            'x':x_t, 
            'y':y_t,
            'phi':phi_t,
            'tau': tau_t,
            'log_q_x':log_q_t_x_t}

    return carry_t

def init_gradients(carry_m1, input_0, p:HMM, q:BackwardSmoother, h_0, num_samples):


    y_0 = input_0['y']
    key_0, unformatted_phi_0 = input_0['key'], input_0['phi']

    theta = carry_m1['theta']

    h = partial(h_0, models={'p':p, 'q':q})

    def h_bar(unformatted_phi_0):

        phi_0 = q.format_params(unformatted_phi_0)
        state_0 = q.init_state(y_0, phi_0)
        filt_params = q.filt_params_from_state(state_0, phi_0)
        x_0 = jax.vmap(q.filt_dist.sample)(jax.random.split(key_0, num_samples), filt_params)
        log_q_x_0 = q.filt_dist.logpdf(x_0, filt_params)

        data_0 = {'x':x_0,
                'log_q_x':log_q_x_0,
                'state':state_0,
                'phi':phi_0,
                'y':y_0}

        data = {'tm1': carry_m1, 't':data_0}

        return named_vmap(h, axes_names={'t':{'x':0}}, input_dict=data), state_0

    jac_Omega_0, s_0 = jax.jacrev(h_bar, has_aux=True)(unformatted_phi_0)
    Omega_0, _ = h_bar(unformatted_phi_0)

    carry = {'stats':{'Omega':Omega_0, 'jac_Omega_0':jac_Omega_0}, 
            's':s_0, 
            'theta':theta,
            'key':key_0}

    return carry

def update_gradient_IS(
        carry_tm1, 
        input_t:HMM, 
        p:HMM, 
        q:BackwardSmoother, 
        h, 
        num_samples, 
        normalizer):

    h = partial(h, models={'p':p, 'q':q})

    key_t, y_t, unformatted_phi_t = input_t['key'], input_t['y'], input_t['phi']

    key_tm1, s_tm1, stats_tm1, theta = carry_tm1['key'], carry_tm1['s'], carry_tm1['stats'], carry_tm1['theta']

    Omega_tm1 = stats_tm1['Omega'] # d-dimensional
    jac_Omega_tm1 = stats_tm1['jac_Omega'] # N x d dimensional

    def w_bar(unformatted_phi, key_t):
        phi = q.format_params(unformatted_phi)
        s_t = q.new_state(y_t, s_tm1, phi)
        params_q_t = q.filt_params_from_state(s_t, phi)
        params_q_tm1 = q.filt_params_from_state(s_tm1, phi)
        params_q_tm1_t = q.backwd_params_from_state(s_tm1, phi)
        x_tm1 = jax.vmap(q.filt_dist.sample, in_axes=(0,None))(random.split(key_tm1, num_samples), params_q_tm1)
        x_t = q.filt_dist.sample(key_t, params_q_t)

        return normalizer(jax.vmap(lambda x_tm1: q.backwd_kernel.logpdf(x_tm1, x_t, params_q_tm1_t) \
                                                            - q.filt_dist.logpdf(x_tm1, params_q_tm1))(x_tm1)), s_t


    def h_bar(unformatted_phi, key_t):
        
        phi = q.format_params(unformatted_phi)
        s_t = q.new_state(y_t, s_tm1, phi)
        params_q_t = q.filt_params_from_state(s_t, phi)
        params_q_tm1 = q.filt_params_from_state(s_tm1, phi)
        params_q_tm1_t = q.backwd_params_from_state(s_tm1, phi)
        x_tm1 = jax.vmap(q.filt_dist.sample, in_axes=(0,None))(random.split(key_tm1, num_samples), params_q_tm1)
        x_t = q.filt_dist.sample(key_t, params_q_t)

        data_t = {
            'x':x_t, 
            'params_filt':params_q_t, 
            'y':y_t, 
            'theta':theta}

        data_tm1 = {
            'x':x_tm1, 
            'params_filt':params_q_tm1,
            'params_backwd':params_q_tm1_t}
            
        return named_vmap(lambda data_tm1:h({'tm1':data_tm1, 't':data_t}), axes_names={'x':0}, input_dict=data_tm1)
        

    def update(key_t):
            
        weights, weights_vjp, s_t = jax.vjp(partial(w_bar, key_t=key_t), unformatted_phi_t, has_aux=True)
        h_t, h_vjp = jax.vjp(partial(h_bar, key_t=key_t), unformatted_phi_t)
        unweighted_term = Omega_tm1 + h_t
        Omega_t = weights @ unweighted_term
        w_t_jac_Omega_tm1 = tree_map(lambda x: x.T @ weights, jac_Omega_tm1)

        jac_Omega_t = tree_map(lambda x,y,z: x + y + z, 
                            weights_vjp(unweighted_term)[0], h_vjp(weights)[0], w_t_jac_Omega_tm1)

        return Omega_t, jac_Omega_t, s_t
                    

    Omega_t, jac_Omega_t, s_t = jax.vmap(update)(random.split(key_t, num_samples))

    s_t = tree_get_idx(0, s_t)

    carry_t = {'stats':{'Omega':Omega_t, 
                        'jac_Omega':jac_Omega_t}, 
                's':s_t,
                'key':key_t,
                'theta':theta}

    return carry_t
    
def update_PaRIS(carry_tm1, 
                input_t:HMM, 
                p:HMM, 
                q:BackwardSmoother, 
                h, 
                num_samples, 
                normalizer, 
                num_paris_samples):


    state_tm1 = carry_tm1['state']
    phi_tm1 = carry_tm1['phi']
    phi_t = input_t['phi']

    params_q_tm1_t = q.backwd_params_from_state(state_tm1, phi_tm1)

    key_t, y_t = input_t['key'], input_t['y']


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
        
    state_t = q.new_state(y_t, state_tm1, phi_t)

    key_q_t, *keys_ancestors = jax.random.split(key_t, num_samples + 1)

    x_t, log_q_t_x_t = samples_and_log_probs(q.filt_dist, 
                                            key_q_t, 
                                            q.filt_params_from_state(state_t, phi_t),
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
            'phi':phi_t,
            'tau':tau_t,
            'y':y_t}

    return carry_t


OnlineNormalizedISELBO = lambda p, q, num_samples: OnlineVariationalAdditiveSmoothing(          
                                                    p, 
                                                    q,
                                                    init_standard,
                                                    update_IS,
                                                    online_elbo_functional(p,q),
                                                    exp_and_normalize,
                                                    num_samples)


OnlineISELBO = lambda p, q, num_samples: OnlineVariationalAdditiveSmoothing(          
                                                    p, 
                                                    q,
                                                    init_standard,
                                                    update_IS,
                                                    online_elbo_functional(p,q),
                                                    None,
                                                    num_samples)

OnlineParisELBO = lambda p, q, num_samples: OnlineVariationalAdditiveSmoothing(          
                                            p, 
                                            q,
                                            init_standard,
                                            partial(update_PaRIS, num_paris_samples=2),
                                            online_elbo_functional(p,q),
                                            exp_and_normalize,
                                            num_samples)