import jax
from jax import vmap, lax, numpy as jnp
from jax.flatten_util import ravel_pytree as ravel
from .stats.hmm import *
from .utils import *
from backward_ica.stats import BackwardSmoother
from backward_ica.variational import NeuralBackwardSmoother


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
        return tree_map(lambda x:  jnp.mean(x, axis=0) / (T+1), stats), outputs

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

def init_PaRIS(
        carry_m1, 
        input_0, 
        p:HMM, 
        q:BackwardSmoother, 
        h_0, 
        num_samples):


    y_0 = input_0['y']
    key_0, unformatted_phi_0 = input_0['key'], input_0['phi']

    phi_0 = q.format_params(unformatted_phi_0)

    state_0 = q.init_state(y_0, phi_0)

    filt_params = q.filt_params_from_state(state_0, phi_0)
    x_0, log_q_x_0 = samples_and_log_probs(q.filt_dist, 
                                            key_0, 
                                            filt_params, 
                                            num_samples,
                                            stop_grad=False)

    # x_0 = jnp.zeros((num_samples, p.state_dim))
    # log_q_x_0 = jax.vmap(q.filt_dist.logpdf, in_axes=(0,None))(x_0, filt_params)
    
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

    
    return carry, x_0

def update_PaRIS(
        carry_tm1, 
        input_t:HMM, 
        p:HMM, 
        q:NeuralBackwardSmoother, 
        h, 
        num_samples, 
        normalizer):


    state_tm1 = carry_tm1['state']


    t, key_t, y_t, unformatted_phi_t = input_t['t'], input_t['key'], input_t['y'], input_t['phi']

    phi_t = q.format_params(unformatted_phi_t)
    state_t = q.new_state(y_t, state_tm1, phi_t)


    filt_params_tm1 = jax.lax.stop_gradient(q.filt_params_from_state(state_tm1, phi_t))
    filt_params_t = q.filt_params_from_state(state_t, phi_t)

    # x_t, log_q_t_x_t = samples_and_log_probs(q.filt_dist, 
    #                                         key_t, 
    #                                         filt_params_t,
    #                                         num_samples,
    #                                         stop_grad=False)

    x_t = jnp.zeros((num_samples, p.state_dim))
    log_q_t_x_t = jax.vmap(q.filt_dist.logpdf, in_axes=(0,None))(x_t, filt_params_t)


    params_q_tm1_t = q.backwd_params_from_state(filt_params_tm1, filt_params_t, phi_t)


    h = partial(h, models={'p':p, 'q':q})

    tau_tm1 = carry_tm1['stats']['tau']
    x_tm1 = carry_tm1['x']
    log_q_tm1_x_tm1 = carry_tm1['log_q_x']

    def update(x_t, log_q_t_x_t):

        data_t = {'x':x_t,'log_q_x':log_q_t_x_t, 'y':y_t}

        def log_weights(x_tm1):
            return q.log_transition_function(x_tm1, x_t, params_q_tm1_t)

            # return q.backwd_kernel.logpdf(x_tm1, x_t, params_q_tm1_t) \
            #     - q.filt_dist.logpdf(x_tm1, filt_params_tm1)#, eta1, eta2
        
        def _h(x_tm1, log_q_tm1_x_tm1):
            data_tm1 = {'x':x_tm1, 
                        'log_q_x':log_q_tm1_x_tm1, 
                        'log_q_backwd_x': q.backwd_kernel.logpdf(x_tm1, x_t, params_q_tm1_t), 
                        'theta':carry_tm1['theta']}
            data = {'t':data_t, 'tm1':data_tm1}
            return h(data)
        
        log_w_tm1_t = jax.vmap(log_weights)(x_tm1)

        w_tm1_t = normalizer(log_w_tm1_t)

        h_tm1_t = jax.vmap(_h)(x_tm1, log_q_tm1_x_tm1)

        result = (w_tm1_t.reshape(-1,1).T @ (tau_tm1 + h_tm1_t).reshape(-1,1)).squeeze()

        # result = tau_tm1 + h_tm1_t
        return result, 1 / (w_tm1_t**2).sum(), jnp.exp(w_tm1_t).sum(), log_w_tm1_t
    

    tau_t, ess_t, normalizing_const, log_weights = update(x_t[0], log_q_t_x_t[0])

    tau_t = jnp.ones((num_samples,)) * tau_t
    ess_t = jnp.ones((num_samples,)) * ess_t
    normalizing_const = jnp.ones((num_samples,)) * normalizing_const
    log_weights = jnp.ones((num_samples,)) * log_weights

    # tau_t, ess_t, normalizing_const, log_weights = update)(x_t, log_q_t_x_t)

    carry_t = {'state':state_t, 
            'x':x_t, 
            'stats': {'tau':tau_t},
            'log_q_x':log_q_t_x_t}
    
    return carry_t, x_t


def init_carry_gradients_reparam(unformatted_params, state_dim, obs_dim, num_samples, out_shape, dummy_state):

    dummy_key = jax.random.PRNGKey(0)
    dummy_Omega = jnp.empty((num_samples, *out_shape))
    dummy_jac_Omega = jax.jacrev(lambda phi:dummy_Omega)(unformatted_params)

    carry = {'key':dummy_key, 
            'stats':{'Omega':dummy_Omega, 'jac_Omega':dummy_jac_Omega},
            's': dummy_state}

    return carry

def init_gradients_reparam(carry_m1, input_0, p:HMM, q:BackwardSmoother, h_0, num_samples):


    key_0, y_0, unformatted_phi_0 = input_0['key'], input_0['y'], input_0['phi']

    h = partial(h_0, models={'p':p, 'q':q})

    def h_bar(unformatted_phi_0, key):

        phi_0 = q.format_params(unformatted_phi_0)
        s_0 = q.init_state(y_0, phi_0)
        filt_params = q.filt_params_from_state(s_0, phi_0)

        x_0 = q.filt_dist.sample(key, filt_params)

        log_q_x_0 = q.filt_dist.logpdf(x_0, filt_params)

        data_0 = {'x':x_0,
                'log_q_x':log_q_x_0,
                'y':y_0}

        data = {'tm1': carry_m1, 't':data_0}

        return h(data), s_0

    (Omega_0, s_0), jac_Omega_0  = jax.vmap(jax.value_and_grad(h_bar, argnums=0, has_aux=True), in_axes=(None, 0))(
                                                                            unformatted_phi_0, 
                                                                            jax.random.split(key_0, num_samples))

    carry = {'stats':{'Omega':Omega_0, 'jac_Omega':jac_Omega_0}, 
            'key':key_0,
            's':tree_get_idx(0, s_0)}

    return carry, None

def update_gradients_reparam(
        carry_tm1, 
        input_t:HMM, 
        p:HMM, 
        q:BackwardSmoother, 
        h, 
        num_samples, 
        normalizer):

    h = partial(h, models={'p':p, 'q':q})

    t, key_t, y_t, unformatted_phi_t = input_t['t'], input_t['key'], input_t['y'], input_t['phi']

    base_key_tm1, stats_tm1, s_tm1, theta = carry_tm1['key'], carry_tm1['stats'], carry_tm1['s'], carry_tm1['theta']

    Omega_tm1 = stats_tm1['Omega'] # d-dimensional
    jac_Omega_tm1 = stats_tm1['jac_Omega'] # N x d dimensional

    def _w(unformatted_phi, index, key_t):
        phi = q.format_params(unformatted_phi)
        s_tm1 = q.get_state(t-1, input_t['ys'], phi)
        s_t = q.new_state(y_t, s_tm1, phi)
        params_q_t = q.filt_params_from_state(s_t, phi)
        params_q_tm1 = q.filt_params_from_state(s_tm1, phi)
        params_q_tm1_t = q.backwd_params_from_state(params_q_tm1, params_q_t, phi)
        x_tm1 = jax.vmap(q.filt_dist.sample, in_axes=(0,None))(random.split(base_key_tm1, num_samples), params_q_tm1)
        x_t = q.filt_dist.sample(key_t, params_q_t)

        return normalizer(jax.vmap(lambda x_tm1: q.log_transition_function(x_tm1, x_t, params_q_tm1_t))(x_tm1))[index], s_t

    def _h(unformatted_phi, key_tm1, key_t):
        
        phi = q.format_params(unformatted_phi)
        s_tm1 = q.get_state(t-1, input_t['ys'], phi)
        s_t = q.new_state(y_t, s_tm1, phi)
        params_q_t = q.filt_params_from_state(s_t, phi)
        params_q_tm1 = q.filt_params_from_state(s_tm1, phi)
        params_q_tm1_t = q.backwd_params_from_state(params_q_tm1, params_q_t, phi)
        x_tm1 = q.filt_dist.sample(key_tm1, params_q_tm1)
        x_t = q.filt_dist.sample(key_t, params_q_t)

        log_q_t = q.filt_dist.logpdf(x_t, params_q_t)
        log_q_tm1 = q.filt_dist.logpdf(x_tm1, params_q_tm1)

        data_t = {
            'x':x_t, 
            'log_q_x':log_q_t, 
            'y':y_t, 
            'theta':theta}

        data_tm1 = {
            'theta':theta,
            'x':x_tm1, 
            'log_q_x':log_q_tm1,
            'log_q_backwd_x':q.backwd_kernel.logpdf(x_tm1, x_t, params_q_tm1_t)}
            
        return h(data={'tm1':data_tm1, 't':data_t})
        


    def update(key_t):
            
        (w_t, s_t), grad_w_t = jax.vmap(jax.value_and_grad(_w, argnums=0, has_aux=True), in_axes=(None, 0, None))(unformatted_phi_t, 
                                                                        jnp.arange(0,num_samples), 
                                                                        key_t)

        h_t, grad_h_t = jax.vmap(jax.value_and_grad(_h, argnums=0), in_axes=(None, 0, None))(unformatted_phi_t, 
                                                                        random.split(base_key_tm1, num_samples), 
                                                                        key_t)
        unweighted_term = Omega_tm1 + h_t
        
        Omega_t = w_t.reshape(-1,1).T @ unweighted_term.reshape(-1,1)

        def sum_of_grads(grad_w_t, w_t, unweighted_term_t, jac_Omega_tm1, grad_h_t):

            return tree_map(lambda x,y,z: (x * unweighted_term_t + w_t * (y+z)).T, grad_w_t, jac_Omega_tm1, grad_h_t)
        
        all_terms_grad  = jax.vmap(sum_of_grads)(grad_w_t, w_t, unweighted_term, jac_Omega_tm1, grad_h_t)

        return Omega_t.squeeze(), tree_map(lambda x: jnp.sum(x, axis=0).squeeze(), all_terms_grad), tree_get_idx(0,s_t)
                    

    Omega_t, jac_Omega_t, s_t = jax.vmap(update)(random.split(key_t, num_samples))


    carry_t = {'stats':{'Omega':Omega_t, 
                        'jac_Omega':jac_Omega_t}, 
                'key':key_t,
                's':tree_get_idx(0, s_t)}

    return carry_t, None




OnlineELBO = lambda p, q, num_samples: OnlineVariationalAdditiveSmoothing(          
                                                    p, 
                                                    q,
                                                    init_carry,
                                                    init_PaRIS,
                                                    update_PaRIS,
                                                    online_elbo_functional(p,q),
                                                    exp_and_normalize,
                                                    num_samples)

OnlineELBOAndGrad = lambda p,q, num_samples: OnlineVariationalAdditiveSmoothing(p, 
                                                                                q, 
                                                                                init_carry_gradients_reparam, 
                                                                                init_gradients_reparam,
                                                                                update_gradients_reparam,
                                                                                online_elbo_functional(p,q),
                                                                                exp_and_normalize,
                                                                                num_samples)