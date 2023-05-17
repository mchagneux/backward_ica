import jax
from jax import vmap, lax, numpy as jnp
from jax.flatten_util import ravel_pytree as ravel
from .stats.hmm import *
from .utils.misc import *
from src.stats import BackwardSmoother
from src.variational import NeuralBackwardSmoother
vmap_ravel = jax.vmap(lambda x: ravel_pytree(x)[0])
import blackjax
def value_and_jac(fun, argnums=0, has_aux=False):
    if has_aux:
        return lambda *args: (fun(*args), jax.jacobian(fun, argnums=argnums, has_aux=True)(*args)[0])
    else: 
        return lambda *args: (fun(*args), jax.jacobian(fun, argnums=argnums, has_aux=False)(*args))



class OnlineVariationalAdditiveSmoothing:

    def __init__(self, 
                p:HMM, 
                q:BackwardSmoother, 
                additive_functional, 
                preprocess_fn,
                init_carry_fn,
                init_fn, 
                update_fn, 
                postprocess_fn=lambda x, **kwargs:x, 
                **options):

        self.p = p
        self.q = q

        self.additive_functional:AdditiveFunctional = additive_functional(p,q)

        self.options = options
        self.options['normalizer'] = exp_and_normalize


        self._preprocess_fn = preprocess_fn
        self._init_carry_fn = init_carry_fn
        self._init_fn = init_fn
        self._update_fn = update_fn
        self._postprocess_fn = postprocess_fn

    def _init(self, carry, input):
        return self._init_fn(carry, input, 
                             p=self.p, 
                             q=self.q, 
                             h=self.additive_functional.init, 
                             **self.options)
        
    def _update(self, carry, input):
        return self._update_fn(carry, input, 
                               p=self.p,
                               q=self.q, 
                               h=self.additive_functional.update, 
                               **self.options)
    
    def init_carry(self, params):

        return self._init_carry_fn(params, 
                                   p=self.p,
                                   q=self.q,
                                   h=self.additive_functional, 
                                   **self.options)
    
    def step(self, carry, input):

        carry, output = lax.cond(input['t'] != 0, 
                        self._update, 
                        self._init,
                        carry, input)
        
        return carry, output
    
    def preprocess(self, obs_seq):
        return self._preprocess_fn(obs_seq, **self.options)

    def batch_compute(self, key, obs_seq, theta, phi):


        T = len(obs_seq) - 1 # T + 1 observations

        keys = jax.random.split(key, T+1) # T+1 keys 
        timesteps = jnp.arange(0, T+1) # [0:T]

        
        def _step(carry, x):
            t, key_t, strided_ys = x
            input_t = {'t':t, 
                       'key':key_t, 
                       'T': T, 
                       'ys_bptt':strided_ys, 
                       'phi':phi}            
            carry['theta'] = theta
            carry_t, output_t = self.step(carry, input_t)
            return carry_t, output_t
        

        carry_m1 = self.init_carry(phi)

        # obs_seq_with_dummy_obs = tree_append(obs_seq, obs_seq[-1])
        # obs_seq_strides = tree_get_strides(2, obs_seq)

        # choices = carry['choices']

        # def mean_on_choices(array, choices):
        #     zeros_array = jnp.zeros_like(array)
        #     return jnp.sum(jnp.where(choices, array, zeros_array), axis=0) / choices.sum()

        # return tree_map(lambda x:  jnp.mean(jnp.exp(carry['log_K']) * x, axis=0) / (T+1), stats)['tau'], outputs
        
    
        strided_ys = self.preprocess(obs_seq)

        return lax.scan(_step, 
                        init=carry_m1,
                        xs=(timesteps, keys, strided_ys))

        # return tree_map(lambda x: mean_on_choices(x, choices) / (T + 1), stats), outputs
        # weights = self.normalizer(carry['log_q_x'] - carry['log_nu_x'])
        # return tree_map(lambda x: (weights.reshape(-1,1).T @ x).squeeze() / (T + 1), stats), outputs

    def postprocess(self, carry, **kwargs):
        return self._postprocess_fn(carry, **kwargs, **self.options)


def init_carry(unformatted_params, **kwargs):


    num_samples = kwargs['num_samples']
    out_shape = kwargs['h'].out_shape
    state_dim = kwargs['p'].state_dim
    dummy_state = kwargs['q'].empty_state()

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
        **kwargs):


    y_0 = input_0['y']
    key_0, unformatted_phi_0 = input_0['key'], input_0['phi']

    p = kwargs['p']
    q = kwargs['q']
    num_samples = kwargs['num_samples']
    h_0 = kwargs['h']

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

    
    return carry, (log_q_x_0, log_q_x_0, filt_params.eta1, jnp.diagonal(filt_params.eta2), jnp.empty((num_samples,num_samples)))

def update_PaRIS(
        carry_tm1, 
        input_t:HMM, 
        **kwargs):

    p:HMM = kwargs['p']
    q:BackwardSmoother = kwargs['q']
    num_samples = kwargs['num_samples']
    normalizer = kwargs['normalizer']
    h = kwargs['h']

    state_tm1 = carry_tm1['state']


    t, key_t, y_t, unformatted_phi_t = input_t['t'], input_t['key'], input_t['y'], input_t['phi']

    phi_t = q.format_params(unformatted_phi_t)
    state_t = q.new_state(y_t, state_tm1, phi_t)


    filt_params_tm1 = q.filt_params_from_state(state_tm1, phi_t)
    filt_params_t = q.filt_params_from_state(state_t, phi_t)

    x_t, log_q_t_x_t = samples_and_log_probs(q.filt_dist, 
                                            key_t, 
                                            filt_params_t,
                                            num_samples)


    states = (state_tm1, state_t)
    params_q_tm1_t = q.backwd_params_from_states(states, phi_t)


    h = partial(h, models={'p':p, 'q':q})

    tau_tm1 = carry_tm1['stats']['tau']
    x_tm1 = carry_tm1['x']
    log_q_tm1_x_tm1 = carry_tm1['log_q_x']

    def update(x_t, log_q_t_x_t):

        data_t = {'x':x_t,'log_q_x':log_q_t_x_t, 'y':y_t}

        def log_weights(x_tm1):
            # return q.log_transition_function(x_tm1, x_t, params_q_tm1_t)

            return q.backwd_kernel.logpdf(x_tm1, x_t, params_q_tm1_t) \
                - q.filt_dist.logpdf(x_tm1, filt_params_tm1)#, eta1, eta2
        
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

        return w_tm1_t @ (tau_tm1 + h_tm1_t), 1 / (w_tm1_t**2).sum(), jnp.exp(w_tm1_t).sum(), log_w_tm1_t
    


    tau_t, ess_t, normalizing_const, log_weights = jax.vmap(update)(x_t, log_q_t_x_t)

    carry_t = {'state':state_t, 
            'x':x_t, 
            'stats': {'tau':tau_t},
            'log_q_x':log_q_t_x_t}
    
    return carry_t, (ess_t, normalizing_const, filt_params_t.eta1, jnp.diagonal(filt_params_t.eta2), log_weights)

def postprocess_PaRIS(carry, **kwargs):
    T = kwargs['T']
    return jnp.mean(carry['stats']['tau'], axis=0) / (T + 1)

def init_carry_gradients_reparam(
            unformatted_params, 
            state_dim, 
            obs_dim, 
            num_samples, 
            out_shape, 
            dummy_state):

    dummy_key = jax.random.PRNGKey(0)
    dummy_tau = jnp.empty((num_samples, *out_shape))
    dummy_grad_tau = jax.jacrev(lambda phi:dummy_tau)(unformatted_params)

    carry = {'key':dummy_key, 
            'stats':{'tau':dummy_tau, 'grad_tau':dummy_grad_tau},
            's': dummy_state}

    return carry

def init_gradients_reparam(
                carry_m1, 
                input_0, 
                p:HMM, 
                q:BackwardSmoother, 
                h_0, 
                num_samples):


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

    (tau_0, s_0), grad_tau_0  = jax.vmap(jax.value_and_grad(h_bar, argnums=0, has_aux=True), in_axes=(None, 0))(
                                                                            unformatted_phi_0, 
                                                                            jax.random.split(key_0, num_samples))

    carry = {'stats':{'tau':tau_0, 'grad_tau':grad_tau_0}, 
            'key':key_0,
            's':tree_get_idx(0, s_0)}

    return carry, 0.0

def update_gradients_reparam(
        carry_tm1, 
        input_t:HMM, 
        p:HMM, 
        q:BackwardSmoother, 
        h, 
        num_samples, 
        normalizer):


    h = partial(h, models={'p':p, 'q':q})

    t, key_t, y_t, unformatted_phi_t = input_t['t'], input_t['key'], \
                        input_t['y'], input_t['phi']

    unravel_grad = ravel_pytree(unformatted_phi_t)[1]

    base_key_tm1, stats_tm1, s_tm1, theta = carry_tm1['key'], carry_tm1['stats'], \
                                        carry_tm1['s'], carry_tm1['theta']

    tau_tm1 = stats_tm1['tau'] # d-dimensional
    grad_tau_tm1 = vmap_ravel(stats_tm1['grad_tau']) # N x d dimensional

    def _w(unformatted_phi, index, key_t):
        phi = q.format_params(unformatted_phi)
        # s_tm1 = q.get_state(t-1, input_t['ys'], phi)
        s_t = q.new_state(y_t, s_tm1, phi)
        params_q_t = q.filt_params_from_state(s_t, phi)
        params_q_tm1 = q.filt_params_from_state(s_tm1, phi)
        params_q_tm1_t = q.backwd_params_from_states((s_tm1,s_t), phi)
        x_tm1 = jax.vmap(q.filt_dist.sample, in_axes=(0,None))(
                                                            random.split(base_key_tm1, num_samples), 
                                                            params_q_tm1)
        x_t = q.filt_dist.sample(key_t, params_q_t)

        return normalizer(jax.vmap(lambda x_tm1: q.backwd_kernel.logpdf(x_tm1, x_t, params_q_tm1_t) \
                - q.filt_dist.logpdf(x_tm1, params_q_tm1))(x_tm1))[index], s_t

    def _h(unformatted_phi, key_tm1, key_t):
        
        phi = q.format_params(unformatted_phi)
        # s_tm1 = q.get_state(t-1, input_t['ys'], phi)
        s_t = q.new_state(y_t, s_tm1, phi)
        params_q_t = q.filt_params_from_state(s_t, phi)
        params_q_tm1 = q.filt_params_from_state(s_tm1, phi)
        params_q_tm1_t = q.backwd_params_from_states((s_tm1,s_t), phi)
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
            
        (w_t, s_t), grad_w_t = jax.vmap(jax.value_and_grad(_w, argnums=0, has_aux=True), 
                                            in_axes=(None, 0, None))(
                                                            unformatted_phi_t, 
                                                            jnp.arange(0,num_samples), 
                                                            key_t)

        grad_w_t = vmap_ravel(grad_w_t)

        h_t, grad_h_t = jax.vmap(jax.value_and_grad(_h, argnums=0), in_axes=(None, 0, None))(unformatted_phi_t, 
                                                                        random.split(base_key_tm1, num_samples), 
                                                                        key_t)
        
        grad_h_t = vmap_ravel(grad_h_t)

        tau_t_unweighted = tau_tm1 + h_t
        
        tau_t = jnp.sum(jax.vmap(lambda w,tau: w*tau)(w_t, tau_t_unweighted), axis=0)
        

        grad_tau_t = jnp.sum(jax.vmap(lambda w, grad_tau, grad_h, grad_w, tau_unweighted: \
                                            w * (grad_tau + grad_h) + tau_unweighted * grad_w)
                                    (w_t, grad_tau_tm1, grad_h_t, grad_w_t, tau_t_unweighted), 
                            axis=0)

        

        return tau_t, unravel_grad(grad_tau_t), tree_get_idx(0,s_t)
                    

    tau_t, grad_tau_t, s_t = jax.vmap(update)(random.split(key_t, num_samples))


    carry_t = {'stats':{'tau':tau_t, 
                        'grad_tau':grad_tau_t}, 
                'key':key_t,
                's':tree_get_idx(0, s_t)}

    return carry_t, 0.0






def preprocess_for_bptt(obs_seq, bptt_depth, **kwargs):


    padded_ys = jnp.concatenate([jnp.empty((bptt_depth-1, obs_seq.shape[1])), 
                                 obs_seq])
    strided_ys = tree_get_strides(bptt_depth, padded_ys)

    return strided_ys 
    
def init_carry_score_gradients(unformatted_params, **kwargs):

    num_samples = kwargs['num_samples']
    state_dim = kwargs['p'].state_dim
    dummy_state = kwargs['q'].empty_state()
    out_shape = kwargs['h'].out_shape
    
    dummy_x = jnp.empty((num_samples, state_dim))
    dummy_H = jnp.empty((num_samples, *out_shape))
    dummy_F = jax.jacrev(lambda phi:dummy_H)(unformatted_params)

    carry = {'base_s': dummy_state, 
            'x':dummy_x, 
            'log_q':jnp.empty((num_samples,)),
            'stats':{'H':dummy_H, 
                    'F':dummy_F},
            'grad_log_q':dummy_F}
    

    return carry

def init_score_gradients(carry_m1, input_0, **kwargs):


    y_0 = input_0['ys_bptt'][-1]
    key_0, unformatted_phi_0 = input_0['key'], input_0['phi']

    p:HMM = kwargs['p']
    q:BackwardSmoother = kwargs['q']
    num_samples = kwargs['num_samples']
    h_0 = kwargs['h']

    s_0 = q.init_state(y_0, 
                       q.format_params(unformatted_phi_0))

    def _log_q_0(unformatted_phi, key):
        phi = q.format_params(unformatted_phi)
        params_q_t = q.filt_params_from_state(s_0, phi)
        x_t = q.filt_dist.sample(key, params_q_t)
        x_t = jax.lax.stop_gradient(x_t)
        return q.filt_dist.logpdf(x_t, params_q_t), x_t


    log_q_0, x_0 = jax.vmap(_log_q_0, 
                                    in_axes=(None,0))(unformatted_phi_0, 
                                                        jax.random.split(key_0, num_samples))
    
    h = partial(h_0, models={'p':p, 'q':q})


    data = {'tm1': carry_m1,
            't':{'x':x_0, 'y':y_0, 'log_q_x':log_q_0}}

    H_0 = named_vmap(h, axes_names={'t':{'x':0, 'log_q_x':0}}, input_dict=data)

    F_0 = tree_map(lambda x: jnp.zeros_like(x), carry_m1['stats']['F'])

    carry = {'stats':{'F':F_0, 
                      'H':H_0},
            'base_s':s_0, 
            'x':x_0,
            'log_q':log_q_0,
            'grad_log_q':tree_map(lambda x: jnp.empty_like(x), carry_m1['grad_log_q'])}

    return carry, 0.0

def update_score_gradients(carry_tm1, input_t, **kwargs):


    p:HMM = kwargs['p']
    q:BackwardSmoother = kwargs['q']
    num_samples = kwargs['num_samples']
    paris = kwargs['paris']
    bptt_depth = kwargs['bptt_depth']
    normalizer, variance_reduction = kwargs['normalizer'], kwargs['variance_reduction']

    t, T, key_t, unformatted_phi_t = input_t['t'], input_t['T'], input_t['key'], input_t['phi']
    ys_for_bptt = input_t['ys_bptt']
    y_t = ys_for_bptt[-1]

    x_tm1, base_s_tm1, stats_tm1, theta = carry_tm1['x'], carry_tm1['base_s'], carry_tm1['stats'], carry_tm1['theta']
    log_q_tm1 = carry_tm1['log_q']
    H_tm1 = stats_tm1['H'] 
    F_tm1 = stats_tm1['F']

    

    def get_states(phi):
        if bptt_depth == 1:
            s_t = q.new_state(y_t, base_s_tm1, phi)
            return s_t, (base_s_tm1, s_t)
        
        return q.get_states(t, 
                            base_s_tm1,
                            ys_for_bptt, 
                            phi)
    
    def _log_q_tm1_t(unformatted_phi, x_tm1, x_t):
        phi = q.format_params(unformatted_phi)
        _ , states = get_states(phi)
        params_q_tm1_t = q.backwd_params_from_states(states, phi)
        log_q_tm1_t = q.backwd_kernel.logpdf(x_tm1, 
                                             x_t, 
                                             params_q_tm1_t)
        return log_q_tm1_t
    
    def _log_q_t(unformatted_phi, key):
        phi = q.format_params(unformatted_phi)
        base_s_t, (_, s_t) = get_states(phi)
        
        params_q_t = q.filt_params_from_state(s_t, phi)
        x_t = q.filt_dist.sample(key, params_q_t)
        x_t = jax.lax.stop_gradient(x_t)
        return q.filt_dist.logpdf(x_t, params_q_t), (x_t, base_s_t)
    
    def _log_q_t_and_dummy_grad(unformatted_phi, key):
        log_q_t, (x_t, base_s_t) = _log_q_t(unformatted_phi, key)
        dummy_grad = tree_map(lambda x: jnp.empty_like(x[0]), carry_tm1['grad_log_q'])
        return (log_q_t, (x_t, base_s_t)), dummy_grad
    
    def _log_q_t_and_grad(unformatted_phi, key):
        return jax.value_and_grad(_log_q_t, has_aux=True)(
                                                        unformatted_phi, 
                                                        key)

    def update(key_t):
    
        if paris:
            key_new_sample, key_paris = jax.random.split(key_t, 2)
        else: 
            key_new_sample = key_t

        (log_q_t, (x_t, base_s_t)), grad_log_q_t = lax.cond(t == T, 
                                                    _log_q_t_and_grad,
                                                    _log_q_t_and_dummy_grad, 
                                                    unformatted_phi_t, 
                                                    key_new_sample)
        def _h(x_tm1, log_q_tm1_t, log_q_tm1):
            log_m_t = log_q_tm1_t - log_q_tm1 + log_q_t
            return p.transition_kernel.logpdf(x_t, x_tm1, theta.transition) \
                + p.emission_kernel.logpdf(y_t, x_t, theta.emission) - log_m_t
            
        _vmaped_h = jax.vmap(_h, in_axes=(0,0,0))
        

        if not paris: 

            log_q_tm1_t, grad_log_q_tm1_t = jax.vmap(jax.value_and_grad(_log_q_tm1_t),
                                                    in_axes=(None,0,None))(unformatted_phi_t, 
                                                                        x_tm1, 
                                                                        x_t)
            # print(x_tm1.shape)
            # print(log_w_t.shape)
            # print(log_q_tm1.shape)
            h_t = _vmaped_h(x_tm1, log_q_tm1_t, log_q_tm1)

            if variance_reduction: 
                moving_average_H = jnp.mean(H_tm1 + h_t, axis=0)
            else: 
                moving_average_H = 0.0
            w_t = normalizer(log_q_tm1_t - log_q_tm1)
            H_t = jax.vmap(lambda w, H, h: w * (H+h))(w_t, H_tm1, h_t)

            F_t = tree_map(lambda F, grad_log_backwd: jax.vmap(lambda w, F, H, h, grad_log_backwd: w*(F + grad_log_backwd*(H+h-moving_average_H)))(
                                                        w_t, 
                                                        F, 
                                                        H_tm1,
                                                        h_t, 
                                                        grad_log_backwd), 
                                                        F_tm1, 
                                                        grad_log_q_tm1_t)
            F_t, H_t = tree_map(lambda x: jnp.sum(x, axis=0), F_t), jnp.sum(H_t, axis=0)

 
        else: 

            log_q_tm1_t = jax.vmap(_log_q_tm1_t,
                                in_axes=(None,0,None))(unformatted_phi_t, 
                                                    x_tm1, 
                                                    x_t)
            

            h_t = _vmaped_h(x_tm1, 
                            log_q_tm1_t,
                            log_q_tm1)

            if variance_reduction: 
                moving_average_H = jnp.mean(H_tm1 + h_t, axis=0)
            else: 
                moving_average_H = 0.0
            # log_w_t = jax.vmap(q.log_fwd_potential, 
            #                    in_axes=(0,None,None,None))(x_tm1, 
            #                                                x_t, 
            #                                                (base_s_tm1, base_s_t), 
            #                                                q.format_params(unformatted_phi_t))

            log_w_t = log_q_tm1_t - log_q_tm1

            # backwd_sampler = blackjax.irmh(logprob_fn=lambda i: log_w_t[i], 
            #                                proposal_distribution=lambda key: jax.random.choice(key, a=num_samples))


            # def _backwd_sample_step(state, x):
            #     step_nb, key = x

            #     def _init(state, key):
            #         return backwd_sampler.init(jax.random.choice(key, a=num_samples))
            #     def _step(state, key):
            #         return backwd_sampler.step(key, state)[0]
                
            #     new_state = jax.lax.cond(step_nb > 0, _step, _init, state, key)
            #     return new_state, new_state.position
                
            # backwd_indices = jax.lax.scan(_backwd_sample_step, 
            #                               init=backwd_sampler.init(0), 
            #                               xs=(jnp.arange(2), 
            #                                   jax.random.split(key_backwd_resampling, 2)))[1]
            


            w_t = normalizer(log_w_t)

            backwd_indices = jax.random.choice(key_paris, 
                                               a=num_samples, 
                                               p=w_t, 
                                               shape=(2,))
            
            sub_x_tm1 = x_tm1[backwd_indices]
            sub_H_tm1 = H_tm1[backwd_indices]
            h_t = h_t[backwd_indices]

            grad_log_q_tm1_t = jax.vmap(jax.grad(_log_q_tm1_t),
                                        in_axes=(None,0,None))(unformatted_phi_t, 
                                                            sub_x_tm1, 
                                                            x_t)


            H_t = jnp.mean(sub_H_tm1 + h_t, axis=0)
            F_t = tree_map(lambda F, grad_log_backwd: jax.vmap(lambda F, H, h, grad_log_backwd: F + grad_log_backwd*(H+h-moving_average_H))(
                                                                    F[backwd_indices], 
                                                                    sub_H_tm1,
                                                                    h_t, 
                                                                    grad_log_backwd[backwd_indices]), 
                                                                    F_tm1, 
                                                                    grad_log_q_tm1_t)
            F_t = tree_map(lambda x: jnp.mean(x, axis=0), F_t)

        return F_t, H_t, x_t, base_s_t, log_q_t, grad_log_q_t

    F_t, H_t, x_t, base_s_t, log_q_t, grad_log_q_t = jax.vmap(update)(jax.random.split(key_t, num_samples))

    carry_t = {'stats':{'F':F_t, 'H':H_t},
            'base_s':tree_get_idx(0,base_s_t), 
            'x':x_t,
            'log_q':log_q_t,
            'grad_log_q':grad_log_q_t}
    
    return carry_t, 0.0

def postprocess_score_gradients(carry, T, variance_reduction, **kwargs):
        H_T = carry['stats']['H']

        F_T = carry['stats']['F']
        grad_log_q_T = carry['grad_log_q']

        elbo_T = jnp.mean(H_T, axis=0)
        if variance_reduction:
            moving_average_H = elbo_T
        else: 
            moving_average_H = 0.0
        elbo_T /= (T + 1)
        grad_T = tree_map(lambda grad_log_q, F: \
                        -jnp.mean(jax.vmap(lambda a,b,c: (a-moving_average_H)*b + c)
                                  (H_T, grad_log_q, F), 
                                axis=0) / (T + 1), grad_log_q_T, F_T)
        return -elbo_T, grad_T


OnlineELBO = lambda p, q, num_samples, **options: OnlineVariationalAdditiveSmoothing(          
                                                    p, 
                                                    q,
                                                    online_elbo_functional,
                                                    init_carry,
                                                    init_PaRIS,
                                                    update_PaRIS,
                                                    postprocess_PaRIS,
                                                    num_samples=num_samples,
                                                    **options)

OnlineELBOScoreGradients = lambda p, q, num_samples, **options: OnlineVariationalAdditiveSmoothing(
                                                                p, 
                                                                q, 
                                                                online_elbo_functional,
                                                                preprocess_for_bptt,
                                                                init_carry_score_gradients, 
                                                                init_score_gradients,
                                                                update_score_gradients,
                                                                postprocess_score_gradients,
                                                                num_samples=num_samples, 
                                                                **options)

OnlineELBOScoreAutodiff = lambda p,q, num_samples: OnlineVariationalAdditiveSmoothing(p, 
                                                                                q, 
                                                                                init_carry_gradients_reparam, 
                                                                                init_gradients_reparam,
                                                                                update_gradients_reparam,
                                                                                online_elbo_functional(p,q),
                                                                                exp_and_normalize,
                                                                                num_samples)

# ThreePaRIS = lambda p,q,functional,num_samples: OnlineVariationalAdditiveSmoothing(
#                                                         p, 
#                                                         q, 
#                                                         init_carry_gradients_score, 
#                                                         init_gradients_score, 
#                                                         update_gradients_F, 
#                                                         functional,
#                                                         exp_and_normalize,
#                                                         num_samples)
