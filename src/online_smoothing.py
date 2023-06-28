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
                init_carry_fn,
                init_fn, 
                update_fn, 
                preprocess_fn,
                postprocess_fn,
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
    
    def preprocess(self, obs_seq, **kwargs):
        return self._preprocess_fn(obs_seq, **kwargs, **self.options)

    def batch_compute(self, key, strided_obs_seq, theta, phi):


        T = len(strided_obs_seq) - 1 # T + 1 observations

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

    
        return lax.scan(_step, 
                        init=carry_m1,
                        xs=(timesteps, keys, strided_obs_seq))
    
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


    y_0 = input_0['ys_bptt']
    key_0, unformatted_phi_0 = input_0['key'], input_0['phi']

    p:HMM = kwargs['p']
    q:BackwardSmoother = kwargs['q']
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

    
    return carry, 0.0

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


    t, key_t, y_t, unformatted_phi_t = input_t['t'], input_t['key'], input_t['ys_bptt'], input_t['phi']

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

        return w_tm1_t @ (tau_tm1 + h_tm1_t)
    


    tau_t = jax.vmap(update)(x_t, log_q_t_x_t)

    carry_t = {'state':state_t, 
            'x':x_t, 
            'stats': {'tau':tau_t},
            'log_q_x':log_q_t_x_t}
    
    return carry_t, 0.0

def postprocess_PaRIS(carry, **kwargs):
    T = kwargs['T']
    return -jnp.mean(carry['stats']['tau'], axis=0) / (T + 1)


def preprocess_for_bptt(obs_seq, bptt_depth, **kwargs):


    padded_ys = jnp.concatenate([jnp.empty((bptt_depth-1, obs_seq.shape[1])), 
                                 obs_seq])
    
    strided_ys = tree_get_strides(bptt_depth, padded_ys)

    return strided_ys 
    

    
def init_carry_elbo_score_gradients(unformatted_params, **kwargs):

    num_samples = kwargs['num_samples']
    state_dim = kwargs['p'].state_dim
    dummy_state = kwargs['q'].empty_state()
    out_shape = kwargs['h'].out_shape
    
    dummy_x = jnp.zeros((num_samples, state_dim))
    dummy_H = jnp.zeros((num_samples, *out_shape))
    dummy_F = jax.jacrev(lambda phi:dummy_H)(unformatted_params)

    carry = {'base_s': dummy_state, 
             's':dummy_state,
            'x':dummy_x, 
            'log_q':jnp.zeros((num_samples,)),
            'stats':{'H':dummy_H, 
                    'F':dummy_F},
            'grad_log_q':dummy_F}
    

    return carry

def init_elbo_score_gradients(carry_m1, input_0, **kwargs):


    y_0 = input_0['ys_bptt'][-1]
    key_0, unformatted_phi_0 = input_0['key'], input_0['phi']

    bptt_depth = kwargs['bptt_depth']
    p:HMM = kwargs['p']
    q:BackwardSmoother = kwargs['q']
    num_samples = kwargs['num_samples']

    def get_state(phi):
        if bptt_depth == 1:
            s_t = q.init_state(y_0, phi)
            return s_t, s_t
        
        base_s, (_, s_0) = q.get_states(0, 
                            carry_m1['base_s'],
                            input_0['ys_bptt'], 
                            phi)
        
        return base_s, s_0
    

    def _log_q_0(unformatted_phi, key):
        phi = q.format_params(unformatted_phi)

        base_s, s_0 = get_state(phi)
        params_q_t = q.filt_params_from_state(s_0, phi)
        x_t = q.filt_dist.sample(key, params_q_t)
        x_t = jax.lax.stop_gradient(x_t)
        return q.filt_dist.logpdf(x_t, params_q_t), (x_t, base_s, s_0, params_q_t)


    (log_q_0, (x_0, base_s, s_0, params_q_0)), grad_log_q_0 = jax.vmap(jax.value_and_grad(_log_q_0, argnums=0, has_aux=True), 
                            in_axes=(None,0))(unformatted_phi_0, 
                                                jax.random.split(key_0, num_samples))
    
    base_s, s_0, params_q_0 = tree_get_idx(0, (base_s, s_0, params_q_0))
    theta:HMM.Params = carry_m1['theta']

    def _h(x_0):
        return p.prior_dist.logpdf(x_0, theta.prior) \
            + p.emission_kernel.logpdf(y_0, x_0, theta.emission)
    
    H_0 = jax.vmap(_h)(x_0)

    F_0 = tree_map(lambda x: jnp.zeros_like(x), 
                   carry_m1['stats']['F'])

    carry = {'stats':{'F':F_0, 
                      'H':H_0},
            's':s_0,
            'base_s':base_s, 
            'x':x_0,
            'log_q':log_q_0,
            'grad_log_q':grad_log_q_0}

    return carry, (params_q_0, q.backwd_params_from_states((s_0, s_0), q.format_params(unformatted_phi_0)))

def update_elbo_score_gradients(carry_tm1, input_t, **kwargs):


    p:HMM = kwargs['p']
    q:BackwardSmoother = kwargs['q']
    num_samples = kwargs['num_samples']
    paris = kwargs['paris']
    bptt_depth = kwargs['bptt_depth']
    mcmc = kwargs['mcmc']
    normalizer, variance_reduction = kwargs['normalizer'], kwargs['variance_reduction']

    t, T, key_t, unformatted_phi_t = input_t['t'], input_t['T'], input_t['key'], input_t['phi']
    ys_for_bptt = input_t['ys_bptt']
    y_t = ys_for_bptt[-1]

    x_tm1, base_s_tm1, s_tm1, stats_tm1, theta = carry_tm1['x'], carry_tm1['base_s'], carry_tm1['s'], carry_tm1['stats'], carry_tm1['theta']
    log_q_tm1 = carry_tm1['log_q']

    H_tm1 =  stats_tm1['H']
    
    F_tm1 = stats_tm1['F']
    # F_tm1 = tree_map(lambda x: jnp.zeros_like(x), stats_tm1['F'])

    def get_states(phi):
        if bptt_depth == 1:
            s_t = q.new_state(y_t, s_tm1, phi)
            return s_t, (s_tm1, s_t)
        
        return q.get_states(t, 
                            base_s_tm1,
                            ys_for_bptt, 
                            phi)


    def _log_q_tm1_t(unformatted_phi, x_tm1, x_t):
        phi = q.format_params(unformatted_phi)
        
        _, (s_tm1, s_t) = get_states(phi)
        params_q_tm1_t = q.backwd_params_from_states((s_tm1, s_t), phi)

        log_q_tm1_t = q.backwd_kernel.logpdf(x_tm1, 
                                             x_t, 
                                             params_q_tm1_t)
        return log_q_tm1_t, params_q_tm1_t
    
    def _log_q_t(unformatted_phi, key):
        phi = q.format_params(unformatted_phi)
        base_s_t, (_ , s_t) = get_states(phi)
        
        params_q_t = q.filt_params_from_state(s_t, phi)
        x_t = q.filt_dist.sample(key, params_q_t)
        x_t = jax.lax.stop_gradient(x_t)
        return q.filt_dist.logpdf(x_t, params_q_t), (x_t, base_s_t, s_t, params_q_t)
    
    def _log_q_t_and_dummy_grad(unformatted_phi, key):
        log_q_t, (x_t, base_s_t, s_t, params_q_t) = _log_q_t(unformatted_phi, key)
        dummy_grad = tree_map(lambda x: jnp.zeros_like(x[0]), carry_tm1['grad_log_q'])
        return (log_q_t, (x_t, base_s_t, s_t, params_q_t)), dummy_grad
    
    def _log_q_t_and_grad(unformatted_phi, key):
        return jax.value_and_grad(_log_q_t, has_aux=True)(
                                                        unformatted_phi, 
                                                        key)

    def update(key_t):
    
        if paris:
            key_new_sample, key_paris = jax.random.split(key_t, 2)
        else: 
            key_new_sample = key_t
        

        (log_q_t, (x_t, base_s_t, s_t, params_q_t)), grad_log_q_t = lax.cond(t == T, 
                                                _log_q_t_and_grad,
                                                _log_q_t_and_dummy_grad, 
                                                unformatted_phi_t, 
                                                key_new_sample)
        
        # (log_q_t, x_t), grad_log_q_t = _log_q_t_and_grad(unformatted_phi_t, key_new_sample)

        def _h(x_tm1, log_q_tm1_t):
            return p.transition_kernel.logpdf(x_t, x_tm1, theta.transition) \
                + p.emission_kernel.logpdf(y_t, x_t, theta.emission) - log_q_tm1_t
            
        _vmaped_h = jax.vmap(_h, in_axes=(0,0))
        

        if not paris: 

            (log_q_tm1_t, params_q_tm1_t), grad_log_q_tm1_t = jax.vmap(jax.value_and_grad(_log_q_tm1_t, has_aux=True),
                                                    in_axes=(None,0,None))(unformatted_phi_t, 
                                                                        x_tm1, 
                                                                        x_t)

            h_t = _vmaped_h(x_tm1, log_q_tm1_t)

            log_w_t = log_q_tm1_t - log_q_tm1
            w_t = normalizer(log_w_t)
            H_t = jax.vmap(lambda w, H, h: w * (H+h))(w_t, H_tm1, h_t)
            H_t = jnp.sum(H_t, axis=0)

            if variance_reduction: 
                control_variate = H_t
            else: 
                control_variate = 0.0
            

            # log_w_t = jax.vmap(q.log_fwd_potential, 
            #                    in_axes=(0,None,None))(x_tm1, 
            #                                           x_t, 
            #                                           q.format_params(unformatted_phi_t))


            F_t = tree_map(lambda F, grad_log_backwd: jax.vmap(lambda w, F, H, h, grad_log_backwd: w*(F + grad_log_backwd*(H+h-control_variate)))(
                                                        w_t, 
                                                        F, 
                                                        H_tm1,
                                                        h_t, 
                                                        grad_log_backwd), 
                                                        F_tm1, 
                                                        grad_log_q_tm1_t)
            
            F_t = tree_map(lambda x: jnp.sum(x, axis=0), F_t)

 
        else: 

            log_q_tm1_t, params_q_tm1_t = jax.vmap(_log_q_tm1_t,
                                in_axes=(None,0,None))(unformatted_phi_t, 
                                                    x_tm1, 
                                                    x_t)
            

            
            # log_w_t = jax.vmap(q.log_fwd_potential, 
            #                    in_axes=(0,None,None,None))(x_tm1, 
            #                                                x_t, 
            #                                                (base_s_tm1, base_s_t), 
            #                                                q.format_params(unformatted_phi_t))

            log_w_t = log_q_tm1_t - log_q_tm1
            
            if mcmc:
                backwd_sampler = blackjax.irmh(logprob_fn=lambda i: log_w_t[i], 
                                            proposal_distribution=lambda key: jax.random.choice(key, a=num_samples))


                def _backwd_sample_step(state, x):
                    step_nb, key = x

                    def _init(state, key):
                        return backwd_sampler.init(jax.random.choice(key, a=num_samples))
                    def _step(state, key):
                        return backwd_sampler.step(key, state)[0]
                    
                    new_state = jax.lax.cond(step_nb > 0, _step, _init, state, key)
                    return new_state, new_state.position
                    
                backwd_indices = jax.lax.scan(_backwd_sample_step, 
                                            init=backwd_sampler.init(0), 
                                            xs=(jnp.arange(3), 
                                                jax.random.split(key_paris, 3)))[1][1:]
                
            else:

                backwd_indices = jax.random.choice(key_paris, 
                                                   a=num_samples, 
                                                   p=normalizer(log_w_t), 
                                                   shape=(2,))
            
            sub_x_tm1 = x_tm1[backwd_indices]
            sub_H_tm1 = H_tm1[backwd_indices]

            sub_h_t = _vmaped_h(sub_x_tm1, 
                                log_q_tm1_t[backwd_indices])

            H_t = jnp.mean(sub_H_tm1 + sub_h_t, axis=0)


            if variance_reduction: 
                control_variate = H_t
            else: 
                control_variate = 0.0


            sub_grad_log_q_tm1_t, _ = jax.vmap(jax.grad(_log_q_tm1_t, has_aux=True),
                                        in_axes=(None,0,None))(unformatted_phi_t, 
                                                            sub_x_tm1, 
                                                            x_t)


            F_t = tree_map(lambda F, sub_grad_log_backwd: jax.vmap(lambda F, H, h, grad_log_backwd: F + grad_log_backwd*(H+h-control_variate))(
                                                                    F[backwd_indices], 
                                                                    sub_H_tm1,
                                                                    sub_h_t, 
                                                                    sub_grad_log_backwd), 
                                                                    F_tm1, 
                                                                    sub_grad_log_q_tm1_t)
            F_t = tree_map(lambda x: jnp.mean(x, axis=0), F_t)

        return F_t, H_t, x_t, log_q_t, grad_log_q_t, base_s_t, s_t, params_q_t, tree_get_idx(0,params_q_tm1_t)

    F_t, H_t, x_t, log_q_t, grad_log_q_t, base_s_t, s_t, params_q_t, params_q_tm1_t = jax.vmap(update)(jax.random.split(key_t, num_samples))


    carry_t = {'stats':{'F':F_t, 'H':H_t},
            'base_s':tree_get_idx(0, base_s_t), 
            's':tree_get_idx(0, s_t),
            'x':x_t,
            'log_q':log_q_t,
            'grad_log_q':grad_log_q_t}
    

    return carry_t, (tree_get_idx(0,params_q_t), tree_get_idx(0, params_q_tm1_t))

def postprocess_elbo_score_gradients(carry, 
                                     variance_reduction, 
                                     **kwargs):

        H_T = carry['stats']['H']
        log_q_T = carry['log_q']
        F_T = carry['stats']['F']
        grad_log_q_T = carry['grad_log_q']
        
        elbo = jnp.mean(H_T - log_q_T, axis=0)


        if variance_reduction:
            moving_average_H = elbo
        else: 
            moving_average_H = 0.0

        
        grad = tree_map(lambda grad_log_q, F: \
                        jnp.mean(jax.vmap(lambda a,b,c: (a-moving_average_H)*b + c)
                                (H_T, grad_log_q, F), 
                                axis=0), grad_log_q_T, F_T)
            

        return elbo, grad
 

def init_carry_elbo_score_gradients_3(unformatted_params, **kwargs):

    num_samples = kwargs['num_samples']
    state_dim = kwargs['p'].state_dim
    dummy_state = kwargs['q'].empty_state()
    out_shape = kwargs['h'].out_shape
    
    dummy_x = jnp.zeros((num_samples, state_dim))
    dummy_H = jnp.zeros((num_samples, *out_shape))
    dummy_F = jax.jacrev(lambda phi:dummy_H)(unformatted_params)

    carry = {'base_s': dummy_state, 
            'x':dummy_x, 
            'log_q':jnp.zeros((num_samples,)),
            'stats':{'H':dummy_H},
            'grad_log_q_bar':dummy_F,
            'grad_H_bar':dummy_F}
    
    return carry

def init_elbo_score_gradients_3(carry_m1, input_0, **kwargs):


    y_0 = input_0['ys_bptt'][-1]
    key_0, unformatted_phi_0 = input_0['key'], input_0['phi']

    p:HMM = kwargs['p']
    q:BackwardSmoother = kwargs['q']
    num_samples = kwargs['num_samples']

    keys = jax.random.split(key_0, num_samples)


    def filt_params_state_and_sample(key, unformatted_phi):
        phi = q.format_params(unformatted_phi)
        s_0 = q.init_state(y_0, phi)
        params_q_t = q.filt_params_from_state(s_0, phi)
        x_t = q.filt_dist.sample(key, params_q_t)
        return x_t, s_0, params_q_t
    

    def _log_q_0_bar(unformatted_phi, key):
        x_t, s_0, params_q_t = filt_params_state_and_sample(key, unformatted_phi)
        return q.filt_dist.logpdf(x_t, params_q_t), (x_t, s_0, params_q_t)


    (log_q_0, (x_0, s_0, params_q_0)), grad_log_q_0_bar = jax.vmap(jax.value_and_grad(_log_q_0_bar, argnums=0, has_aux=True),
                                    in_axes=(None,0))(unformatted_phi_0, keys)
    
    s_0 = tree_get_idx(0, s_0)
    params_q_0 = tree_get_idx(0, params_q_0)

    theta:HMM.Params = carry_m1['theta']

    def _h(unformatted_phi, key):
        x_0 = filt_params_state_and_sample(key, unformatted_phi)[0]
        return p.prior_dist.logpdf(x_0, theta.prior) \
            + p.emission_kernel.logpdf(y_0, x_0, theta.emission)
    
    H_0, grad_H_0_bar = jax.vmap(jax.value_and_grad(_h), in_axes=(None,0))(unformatted_phi_0, keys)


    carry = {'stats':{'H':H_0},
            'base_s':s_0, 
            'x':x_0,
            'log_q':log_q_0,
            'grad_log_q_bar':grad_log_q_0_bar,
            'grad_H_bar':grad_H_0_bar}

    return carry, (params_q_0, q.backwd_params_from_states((s_0, s_0), 
                                                           q.format_params(unformatted_phi_0)))

def update_elbo_score_gradients_3(carry_tm1, input_t, **kwargs):


    p:HMM = kwargs['p']
    q:BackwardSmoother = kwargs['q']
    num_samples = kwargs['num_samples']
    paris = kwargs['paris']
    bptt_depth = kwargs['bptt_depth']
    mcmc = kwargs['mcmc']
    normalizer, variance_reduction = kwargs['normalizer'], kwargs['variance_reduction']

    t, T, key_t, unformatted_phi_t = input_t['t'], input_t['T'], \
                                    input_t['key'], input_t['phi']
    ys_for_bptt = input_t['ys_bptt']
    y_t = ys_for_bptt[-1]

    x_tm1, base_s_tm1, stats_tm1, theta = carry_tm1['x'], carry_tm1['base_s'], \
                                        carry_tm1['stats'], carry_tm1['theta']
    log_q_tm1 = carry_tm1['log_q']

    H_tm1 =  stats_tm1['H']

    def sample_and_byproducts(key_t, unformatted_phi):
        phi = q.format_params(unformatted_phi)
        s_t = q.new_state(y_t, base_s_tm1, phi)
        params_q_t = q.filt_params_from_state(s_t, phi)
        x_t = q.filt_dist.sample(key_t, params_q_t)
        return x_t, s_t, phi, params_q_t

    def _log_q_tm1_t_bar(unformatted_phi, x_tm1, key_t):
        x_t, s_t, phi = sample_and_byproducts(key_t, unformatted_phi)[:-1]
        params_q_tm1_t = q.backwd_params_from_states((base_s_tm1, s_t), phi)
        return q.backwd_kernel.logpdf(x_tm1, 
                                    x_t, 
                                    params_q_tm1_t), params_q_tm1_t
    
    def _log_q_t_bar(unformatted_phi, key_t):
        x_t, s_t, _ , params_q_t = sample_and_byproducts(key_t, unformatted_phi)
        return q.filt_dist.logpdf(x_t, params_q_t), (x_t, s_t, params_q_t)


    def _l_theta_bar(unformatted_phi, x_tm1, key_t):
        x_t = sample_and_byproducts(key_t, unformatted_phi)[0]
        l_theta = p.transition_kernel.logpdf(x_t, x_tm1, theta.transition) \
            + p.emission_kernel.logpdf(y_t, x_t, theta.emission)
        return l_theta
            


    def update(key_t):
    
        if paris:
            key_new_sample, key_paris = jax.random.split(key_t, 2)
        else: 
            key_new_sample = key_t


        (log_q_t, (x_t, base_s_t, params_q_t)), grad_log_q_t_bar = jax.value_and_grad(_log_q_t_bar, has_aux=True)(unformatted_phi_t, 
                                                                            key_new_sample)
            
        if not paris: 

        
            (log_q_tm1_t, params_q_tm1_t), grad_log_q_tm1_t_bar = jax.vmap(jax.value_and_grad(_log_q_tm1_t_bar, has_aux=True),
                                                        in_axes=(None,0,None))(unformatted_phi_t, 
                                                                            x_tm1, 
                                                                            key_new_sample)
            

            l_theta, grad_l_theta_bar = jax.vmap(jax.value_and_grad(_l_theta_bar), in_axes=(None, 0, None))(unformatted_phi_t, 
                                                                                                      x_tm1, 
                                                                                                      key_new_sample)

            h_t = l_theta - log_q_tm1_t

            log_w_t = log_q_tm1_t - log_q_tm1

            w_t = normalizer(log_w_t)

            H_t = jax.vmap(lambda w, H, h: w * (H+h))(w_t, H_tm1, h_t)
            H_t = jnp.sum(H_t, axis=0)

            if variance_reduction: 
                control_variate = H_t
            else: 
                control_variate = 0.0
            
            # log_w_t = jax.vmap(q.log_fwd_potential, 
            #                    in_axes=(0,None,None))(x_tm1, 
            #                                           x_t, 
            #                                           q.format_params(unformatted_phi_t))

            grad_H_t_bar = tree_map(lambda grad_l_theta_bar, grad_log_backwd: jax.vmap(lambda w, grad_l_theta_bar, H, h, grad_log_backwd: w*(grad_l_theta_bar + grad_log_backwd*(H+h-control_variate)))(
                                                                                        w_t, 
                                                                                        grad_l_theta_bar, 
                                                                                        H_tm1,
                                                                                        h_t, 
                                                                                        grad_log_backwd), 
                                        grad_l_theta_bar, 
                                        grad_log_q_tm1_t_bar)

            grad_H_t_bar = tree_map(lambda x: jnp.sum(x, axis=0), grad_H_t_bar)

            



        else:
            log_q_tm1_t, params_q_tm1_t = jax.vmap(_log_q_tm1_t_bar, in_axes=(None,0,None))(
                                                        unformatted_phi_t, 
                                                        x_tm1, 
                                                        key_new_sample)
            log_w_t = log_q_tm1_t - log_q_tm1

            w_t = normalizer(log_w_t)
            backwd_indices = jax.random.choice(key_paris, 
                                    a=num_samples, 
                                    p=w_t, 
                                    shape=(2,))
            
            sub_x_tm1 = x_tm1[backwd_indices]
            sub_H_tm1 = H_tm1[backwd_indices]

            sub_l_theta, sub_grad_l_theta_bar = jax.vmap(jax.value_and_grad(_l_theta_bar), 
                                                         in_axes=(None, 0, None))(
                                                                                unformatted_phi_t, 
                                                                                sub_x_tm1, 
                                                                                key_new_sample)


            sub_h_t = sub_l_theta - log_q_tm1_t[backwd_indices]




            H_t = jnp.mean(sub_H_tm1 + sub_h_t, axis=0)


            if variance_reduction: 
                control_variate = H_t
            else: 
                control_variate = 0.0


            sub_grad_log_q_tm1_t = jax.vmap(jax.grad(lambda x,y,z:_log_q_tm1_t_bar(x,y,z)[0]),
                                        in_axes=(None,0,None))(unformatted_phi_t, 
                                                            sub_x_tm1, 
                                                            key_new_sample)
            
            def grad_update(sub_grad_log_q_tm1_t,
                            sub_grad_l_theta_bar, 
                            sub_H_tm1,
                            sub_h_t):

                grad_H_t_bar = tree_map(lambda sub_grad_l_theta_bar, sub_grad_log_backwd: jax.vmap(lambda grad_l_theta_bar, H, h, grad_log_backwd: grad_l_theta_bar + grad_log_backwd*(H+h-control_variate))(
                                                                        sub_grad_l_theta_bar, 
                                                                        sub_H_tm1,
                                                                        sub_h_t, 
                                                                        sub_grad_log_backwd), 
                                                                        sub_grad_l_theta_bar, 
                                                                        sub_grad_log_q_tm1_t)
                
                grad_H_t_bar = tree_map(lambda x: jnp.mean(x, axis=0), grad_H_t_bar)

                return grad_H_t_bar
            
            grad_H_t_bar = grad_update(sub_grad_log_q_tm1_t, sub_grad_l_theta_bar, sub_H_tm1, sub_h_t)

        return grad_H_t_bar, H_t, x_t, log_q_t, grad_log_q_t_bar, base_s_t, params_q_t, tree_get_idx(0, params_q_tm1_t)

    grad_H_t_bar, H_t, x_t, log_q_t, grad_log_q_t_bar, base_s_t, params_q_t, params_q_tm1_t = jax.vmap(update)(jax.random.split(key_t, num_samples))


    carry_t = {'stats':{'H':H_t},
            'base_s':tree_get_idx(0, base_s_t), 
            'x':x_t,
            'log_q':log_q_t,
            'grad_log_q_bar':grad_log_q_t_bar,
            'grad_H_bar':grad_H_t_bar}
    

    return carry_t, (tree_get_idx(0, params_q_t), tree_get_idx(0, params_q_tm1_t)),# A_backwd, a_backwd, Sigma_backwd)

def postprocess_elbo_score_gradients_3(carry, 
                                     **kwargs):



        H_T = carry['stats']['H']
        log_q_T = carry['log_q']
        grad_H_bar_T = carry['grad_H_bar']
        grad_log_q_T = carry['grad_log_q_bar']

        
        elbo = jnp.mean(H_T - log_q_T, axis=0)


        grad = tree_map(lambda x,y: jnp.mean(x-y, axis=0), 
                        grad_H_bar_T, 
                        grad_log_q_T)

        return elbo, grad



def init_carry_elbo_score_gradients_4(unformatted_params, **kwargs):

    num_samples = kwargs['num_samples']
    state_dim = kwargs['p'].state_dim
    dummy_state = kwargs['q'].empty_state()
    out_shape = kwargs['h'].out_shape
    
    dummy_x = jnp.zeros((num_samples, state_dim))
    dummy_H = jnp.zeros((num_samples, *out_shape))
    dummy_F = jax.jacrev(lambda phi:dummy_H)(unformatted_params)

    carry = {'base_s': dummy_state, 
            'x':dummy_x, 
            'log_q':jnp.zeros((num_samples,)),
            'stats':{'H':dummy_H},
            'grad_log_q_bar':dummy_F,
            'grad_H_bar':dummy_F}
    
    return carry

def init_elbo_score_gradients_4(carry_m1, input_0, **kwargs):


    y_0 = input_0['ys_bptt'][-1]
    key_0, unformatted_phi_0 = input_0['key'], input_0['phi']

    p:HMM = kwargs['p']
    q:BackwardSmoother = kwargs['q']
    num_samples = kwargs['num_samples']

    keys = jax.random.split(key_0, num_samples)


    def filt_params_state_and_sample(key, unformatted_phi):
        phi = q.format_params(unformatted_phi)
        s_0 = q.init_state(y_0, phi)
        params_q_t = q.filt_params_from_state(s_0, phi)
        x_t = q.filt_dist.sample(key, params_q_t)
        return x_t, s_0, params_q_t
    

    def _log_q_0_bar(unformatted_phi, key):
        x_t, s_0, params_q_t = filt_params_state_and_sample(key, unformatted_phi)
        return q.filt_dist.logpdf(x_t, params_q_t), (x_t, s_0, params_q_t)


    (log_q_0, (x_0, s_0, params_q_0)), grad_log_q_0_bar = jax.vmap(jax.value_and_grad(_log_q_0_bar, argnums=0, has_aux=True),
                                    in_axes=(None,0))(unformatted_phi_0, keys)
    
    s_0 = tree_get_idx(0, s_0)
    params_q_0 = tree_get_idx(0, params_q_0)

    theta:HMM.Params = carry_m1['theta']

    def _h(unformatted_phi, key):
        x_0 = filt_params_state_and_sample(key, unformatted_phi)[0]
        return p.prior_dist.logpdf(x_0, theta.prior) \
            + p.emission_kernel.logpdf(y_0, x_0, theta.emission)
    
    H_0, grad_H_0_bar = jax.vmap(jax.value_and_grad(_h), in_axes=(None,0))(unformatted_phi_0, keys)


    carry = {'stats':{'H':H_0},
            'base_s':s_0, 
            'x':x_0,
            'log_q':log_q_0,
            'grad_log_q_bar':grad_log_q_0_bar,
            'grad_H_bar':grad_H_0_bar}

    return carry, (params_q_0, q.backwd_params_from_states((s_0, s_0), 
                                                           q.format_params(unformatted_phi_0)))

def update_elbo_score_gradients_4(carry_tm1, input_t, **kwargs):


    p:HMM = kwargs['p']
    q:BackwardSmoother = kwargs['q']
    num_samples = kwargs['num_samples']
    paris = kwargs['paris']
    bptt_depth = kwargs['bptt_depth']
    mcmc = kwargs['mcmc']
    normalizer, variance_reduction = kwargs['normalizer'], kwargs['variance_reduction']

    t, T, key_t, unformatted_phi_t = input_t['t'], input_t['T'], \
                                    input_t['key'], input_t['phi']
    ys_for_bptt = input_t['ys_bptt']
    y_t = ys_for_bptt[-1]

    x_tm1, base_s_tm1, stats_tm1, theta = carry_tm1['x'], carry_tm1['base_s'], \
                                        carry_tm1['stats'], carry_tm1['theta']
    log_q_tm1 = carry_tm1['log_q']

    H_tm1 =  stats_tm1['H']


    formatted_phi, formatting_vjp = jax.vjp(q.format_params, unformatted_phi_t)

    s_t, state_vjp = jax.vjp(lambda formatted_phi:q.new_state(y_t, base_s_tm1, formatted_phi), formatted_phi)

    def _filt_params(formatted_phi):
        return q.filt_params_from_state(s_t, formatted_phi)
    
    params_q_t, params_q_t_vjp = jax.vjp(_filt_params, formatted_phi)
    
    def _x(params_q_t, key_t):
        return q.filt_dist.sample(key_t, params_q_t)
    
    


    def _log_q_t_bar(formatted_phi, key_t):
        s_t = q.new_state(y_t, base_s_tm1, formatted_phi)
        params_q_t = q.filt_params_from_state(s_t, formatted_phi)
        x_t = q.filt_dist.sample(key_t, params_q_t)
        return x_t, q.filt_dist.sample(key_t, params_q_t)
    
    def _log_q_tm1_t_bar(formatted_phi, x_tm1, x_t):
        params_q_tm1_t = q.backwd_params_from_states((base_s_tm1, None), q.format_params(formatted_phi))
        return q.backwd_kernel.logpdf(x_tm1, 
                                    x_t, 
                                    params_q_tm1_t), params_q_tm1_t
    
    def _l_theta_bar(x_tm1, x_t):
        l_theta = p.transition_kernel.logpdf(x_t, x_tm1, theta.transition) \
            + p.emission_kernel.logpdf(y_t, x_t, theta.emission)
        return l_theta
            


    def update(key_t):
    
        if paris:
            key_new_sample, key_paris = jax.random.split(key_t, 2)
        else: 
            key_new_sample = key_t


        (log_q_t, (x_t, base_s_t, params_q_t)), grad_log_q_t_bar = jax.value_and_grad(_log_q_t_bar, has_aux=True)(unformatted_phi_t, 
                                                                            key_new_sample)
            
        if not paris: 

        
            (log_q_tm1_t, params_q_tm1_t), grad_log_q_tm1_t_bar = jax.vmap(jax.value_and_grad(_log_q_tm1_t_bar, has_aux=True),
                                                        in_axes=(None,0,None))(unformatted_phi_t, 
                                                                            x_tm1, 
                                                                            key_new_sample)
            

            l_theta, grad_l_theta_bar = jax.vmap(jax.value_and_grad(_l_theta_bar), in_axes=(None, 0, None))(unformatted_phi_t, 
                                                                                                      x_tm1, 
                                                                                                      key_new_sample)

            h_t = l_theta - log_q_tm1_t

            log_w_t = log_q_tm1_t - log_q_tm1

            w_t = normalizer(log_w_t)

            H_t = jax.vmap(lambda w, H, h: w * (H+h))(w_t, H_tm1, h_t)
            H_t = jnp.sum(H_t, axis=0)

            if variance_reduction: 
                control_variate = H_t
            else: 
                control_variate = 0.0
            
            # log_w_t = jax.vmap(q.log_fwd_potential, 
            #                    in_axes=(0,None,None))(x_tm1, 
            #                                           x_t, 
            #                                           q.format_params(unformatted_phi_t))

            grad_H_t_bar = tree_map(lambda grad_l_theta_bar, grad_log_backwd: jax.vmap(lambda w, grad_l_theta_bar, H, h, grad_log_backwd: w*(grad_l_theta_bar + grad_log_backwd*(H+h-control_variate)))(
                                                                                        w_t, 
                                                                                        grad_l_theta_bar, 
                                                                                        H_tm1,
                                                                                        h_t, 
                                                                                        grad_log_backwd), 
                                        grad_l_theta_bar, 
                                        grad_log_q_tm1_t_bar)

            grad_H_t_bar = tree_map(lambda x: jnp.sum(x, axis=0), grad_H_t_bar)

            
        else:
            log_q_tm1_t, params_q_tm1_t = jax.vmap(_log_q_tm1_t_bar, in_axes=(None,0,None))(
                                                        unformatted_phi_t, 
                                                        x_tm1, 
                                                        key_new_sample)
            log_w_t = log_q_tm1_t - log_q_tm1

            w_t = normalizer(log_w_t)
            backwd_indices = jax.random.choice(key_paris, 
                                    a=num_samples, 
                                    p=w_t, 
                                    shape=(2,))
            
            sub_x_tm1 = x_tm1[backwd_indices]
            sub_H_tm1 = H_tm1[backwd_indices]

            sub_l_theta, sub_grad_l_theta_wrt_x = jax.vmap(jax.value_and_grad(_l_theta_bar, argnums=1),
                                                                in_axes=(None, 0, None))(
                                                                                        unformatted_phi_t, 
                                                                                        sub_x_tm1, 
                                                                                        key_new_sample)


            sub_h_t = sub_l_theta - log_q_tm1_t[backwd_indices]




            H_t = jnp.mean(sub_H_tm1 + sub_h_t, axis=0)


            if variance_reduction: 
                control_variate = H_t
            else: 
                control_variate = 0.0


            sub_grad_log_q_tm1_t = jax.vmap(jax.grad(lambda x,y,z:_log_q_tm1_t_bar(x,y,z)[0]),
                                        in_axes=(None,0,None))(unformatted_phi_t, 
                                                            sub_x_tm1, 
                                                            key_new_sample)
            
            def grad_update(sub_grad_log_q_tm1_t,
                            sub_grad_l_theta_bar, 
                            sub_H_tm1,
                            sub_h_t):

                grad_H_t_bar = tree_map(lambda sub_grad_l_theta_bar, sub_grad_log_backwd: jax.vmap(lambda grad_l_theta_bar, H, h, grad_log_backwd: grad_l_theta_bar + grad_log_backwd*(H+h-control_variate))(
                                                                        sub_grad_l_theta_bar, 
                                                                        sub_H_tm1,
                                                                        sub_h_t, 
                                                                        sub_grad_log_backwd), 
                                                                        sub_grad_l_theta_bar, 
                                                                        sub_grad_log_q_tm1_t)
                
                grad_H_t_bar = tree_map(lambda x: jnp.mean(x, axis=0), grad_H_t_bar)

                return grad_H_t_bar
            
            grad_H_t_bar = grad_update(sub_grad_log_q_tm1_t, sub_grad_l_theta_bar, sub_H_tm1, sub_h_t)

        return grad_H_t_bar, H_t, x_t, log_q_t, grad_log_q_t_bar, base_s_t, params_q_t, tree_get_idx(0, params_q_tm1_t)

    grad_H_t_bar, H_t, x_t, log_q_t, grad_log_q_t_bar, base_s_t, params_q_t, params_q_tm1_t = jax.vmap(update)(jax.random.split(key_t, num_samples))


    carry_t = {'stats':{'H':H_t},
            'base_s':tree_get_idx(0, base_s_t), 
            'x':x_t,
            'log_q':log_q_t,
            'grad_log_q_bar':grad_log_q_t_bar,
            'grad_H_bar':grad_H_t_bar}
    

    return carry_t, (tree_get_idx(0, params_q_t), tree_get_idx(0, params_q_tm1_t)),# A_backwd, a_backwd, Sigma_backwd)

def postprocess_elbo_score_gradients_4(carry, 
                                     **kwargs):



        H_T = carry['stats']['H']
        log_q_T = carry['log_q']
        grad_H_bar_T = carry['grad_H_bar']
        grad_log_q_T = carry['grad_log_q_bar']

        
        elbo = jnp.mean(H_T - log_q_T, axis=0)


        grad = tree_map(lambda x,y: jnp.mean(x-y, axis=0), 
                        grad_H_bar_T, 
                        grad_log_q_T)

        return elbo, grad



OnlineELBO = lambda p, q, num_samples, **options: OnlineVariationalAdditiveSmoothing(          
                                                    p, 
                                                    q,
                                                    online_elbo_functional,
                                                    init_carry_fn=init_carry,
                                                    init_fn=init_PaRIS,
                                                    update_fn=update_PaRIS,
                                                    preprocess_fn=lambda x, **kwargs:x,
                                                    postprocess_fn=postprocess_PaRIS,
                                                    num_samples=num_samples,
                                                    **options)

OnlineELBOScoreGradients = lambda p, q, num_samples, **options: OnlineVariationalAdditiveSmoothing(
                                                                p, 
                                                                q, 
                                                                online_elbo_functional,
                                                                init_carry_fn=init_carry_elbo_score_gradients, 
                                                                init_fn=init_elbo_score_gradients,
                                                                update_fn=update_elbo_score_gradients,
                                                                preprocess_fn=preprocess_for_bptt,
                                                                postprocess_fn=postprocess_elbo_score_gradients,
                                                                num_samples=num_samples, 
                                                                **options)

OnlineELBOScoreTruncatedGradients = lambda p, q, num_samples, **options: OnlineVariationalAdditiveSmoothing(
                                                                p, 
                                                                q, 
                                                                online_elbo_functional,
                                                                init_carry_fn=init_carry_elbo_score_gradients_3, 
                                                                init_fn=init_elbo_score_gradients_3,
                                                                update_fn=update_elbo_score_gradients_3,
                                                                preprocess_fn=preprocess_for_bptt,
                                                                postprocess_fn=postprocess_elbo_score_gradients_3,
                                                                num_samples=num_samples, 
                                                                **options)