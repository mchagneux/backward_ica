import jax
from jax import vmap, lax, numpy as jnp
from .stats.hmm import *
from .utils.misc import *
from src.stats import BackwardSmoother, TwoFilterSmoother

class OfflineVariationalAdditiveSmoothing:

    def __init__(self, p: HMM, q: BackwardSmoother, functional, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples
        self.functional: AdditiveFunctional = functional


    def preprocess(self, data, **kwargs):
        return data 
    
    def __call__(self, key, obs_seq, T, theta: HMM.Params, phi):

        # theta.compute_covs()

        def t_strictly_greater_than_T(carry_tp1, input_t):

            return carry_tp1, 0.0

        def t_smaller_or_equal_to_T(carry_tp1, input_t):

            def t_equals_T(carry_tp1, input_t):

                key_t = input_t['key']
                state_t = input_t['state']

                params_q_t = self.q.filt_params_from_state(state_t, phi)
                x_t = self.q.filt_dist.sample(key_t, params_q_t)

                data_t = {'x':x_t, 'params_q':params_q_t}
            
                tau_t = self.functional.end(models={'p':self.p, 'q':self.q}, 
                                            data={'t':data_t})

                carry_t = {'x':x_t, 'y': input_t['y'], 'tau': tau_t, 'state':state_t}

                return carry_t, 0.0
                

            def t_strictly_lower_than_T(carry_tp1, input_t):

                def t_equals_0(carry_tp1, input_t):


                    x_tp1 = carry_tp1['x']
                    y_tp1 = carry_tp1['y']

                    key_t = input_t['key']
                    state_t = input_t['state']
                    y_t = input_t['y']


                    states = (state_t, carry_tp1['state'])
                    params_q_t_tp1 = self.q.backwd_params_from_states(states, phi)

                    x_t = self.q.backwd_kernel.sample(key_t, x_tp1, params_q_t_tp1)

                    data_tp1 = {'x': x_tp1, 'y':y_tp1, 'theta': theta}

                    data_t = {'x': x_t, 
                            'params_backwd':params_q_t_tp1, 
                            'y': y_t}

                    tau_t = carry_tp1['tau'] + self.functional.init(models={'p':self.p, 'q':self.q}, 
                                                        data={'tp1':data_tp1, 't': data_t})

                    carry_t = {'x':x_t, 'y':input_t['y'], 'tau': tau_t, 'state':state_t}

                    return carry_t, 0.0


                def t_strictly_greater_than_0(carry_tp1, input_t):

                    x_tp1 = carry_tp1['x']
                    y_tp1 = carry_tp1['y']

                    key_t = input_t['key']
                    state_t = input_t['state']


                    states = (state_t, carry_tp1['state'])
                    params_q_t_tp1 = self.q.backwd_params_from_states(states, phi)
                    x_t = self.q.backwd_kernel.sample(key_t, x_tp1, params_q_t_tp1)

                    data_tp1 = {'x': x_tp1, 'y':y_tp1, 'theta': theta}
                    data_t = {'x': x_t, 'params_backwd':params_q_t_tp1}

                    tau_t = carry_tp1['tau'] + self.functional.update(models={'p':self.p, 'q':self.q}, 
                                                    data={'tp1':data_tp1, 't': data_t})

                    carry_t = {'x':x_t, 'y':input_t['y'], 'tau': tau_t, 'state':state_t}

                    return carry_t, 0.0

                return lax.cond(input_t['t'] > 0, 
                                t_strictly_greater_than_0, t_equals_0, 
                                carry_tp1, input_t)

            return lax.cond(input_t['t'] < T, 
                            t_strictly_lower_than_T, t_equals_T, 
                            carry_tp1, input_t)


        def compute(carry_tp1, input_t):

            
            return lax.cond(input_t['t'] <= T, 
                            t_smaller_or_equal_to_T, t_strictly_greater_than_T, 
                            carry_tp1, input_t)


        t_seq = jnp.arange(0, len(obs_seq))

        state_seq = self.q.compute_state_seq(obs_seq, 
                                            compute_up_to=T, 
                                            formatted_params=phi)
        def evaluate_one_path(key):

            inputs = {
                    't': t_seq,
                    'state': state_seq,
                    'y': obs_seq, 
                    'key': jax.random.split(key, len(obs_seq))
                    }

            dummy_carry = {'x': jnp.empty((self.p.state_dim,)),
                            'y': jnp.empty((self.p.obs_dim,)),
                            'tau':jnp.empty((*self.functional.out_shape,)),
                            'state':tree_get_idx(-1,state_seq)}
            
            return lax.scan(compute, 
                            init=dummy_carry, 
                            xs=inputs, 
                            reverse=True)

        carry, aux = jax.vmap(evaluate_one_path)(jax.random.split(key, self.num_samples))
        
        return jnp.mean(carry['tau'], axis=0), 0.0

GeneralBackwardELBO = lambda p, q, num_samples: OfflineVariationalAdditiveSmoothing(p, q, offline_elbo_functional(p,q), num_samples)

class GeneralForwardELBO:

    def __init__(self, p:HMM, q:TwoFilterSmoother, num_samples=200):

        self.p = p
        self.q = q
        self.num_samples = num_samples 


    def preprocess(self, data, **kwargs):
        return data 
    
    def __call__(self, key, obs_seq, compute_up_to, theta, phi):

        def _monte_carlo_sample(key, obs_seq, init_state, backwd_variables_seq):
            
            def _sample_step(prev_sample, x):

                key, obs, idx = x

                def false_fun(key, prev_sample, obs, idx):
                    return prev_sample, 0.0

                def true_fun(key, prev_sample, obs, idx):

                    def init_term(key, prev_sample, obs, idx):
                        init_law_params = self.q.compute_marginal(self.q.init_filt_params(init_state, phi), 
                                                tree_get_idx(0, backwd_variables_seq)) 
                        init_sample = self.q.marginal_dist.sample(key, init_law_params)
                        init_term = self.p.emission_kernel.logpdf(obs, init_sample, theta.emission) \
                                    + self.p.prior_dist.logpdf(init_sample, theta.prior) \
                                    - self.q.marginal_dist.logpdf(init_sample, init_law_params)
                        return init_sample, init_term


                    def other_terms(key, prev_sample, obs, idx):
                        forward_params = self.q.forward_params_from_backwd_var(tree_get_idx(idx, backwd_variables_seq), phi)
                        sample = self.q.forward_kernel.sample(key, prev_sample, forward_params)
                        emission_term_p = self.p.emission_kernel.logpdf(obs, sample, theta.emission)
                        transition_term_p = self.p.transition_kernel.logpdf(sample, prev_sample, theta.transition)
                        fwd_term_q = -self.q.forward_kernel.logpdf(sample, prev_sample, forward_params)
                        return sample,  emission_term_p + transition_term_p + fwd_term_q


                    return lax.cond(idx > 0, other_terms, init_term, key, prev_sample, obs, idx)

                return lax.cond(idx <= compute_up_to, true_fun, false_fun, key, prev_sample, obs, idx)

            terms = lax.scan(f=_sample_step, 
                            init=jnp.empty((self.p.state_dim,)), 
                            xs=(jax.random.split(key, obs_seq.shape[0]), 
                                obs_seq, 
                                jnp.arange(0, len(obs_seq))), 
                                reverse=False)[1]


            return jnp.sum(terms), 0.0


        parallel_sampler = vmap(_monte_carlo_sample, in_axes=(0,None,None,None))

        keys = jax.random.split(key, self.num_samples)

        state_seq = self.q.compute_state_seq(obs_seq, phi)
        backwd_variables_seq = self.q.compute_backwd_variables_seq(state_seq, compute_up_to, phi)


        mc_samples = parallel_sampler(keys, 
                                    obs_seq, 
                                    tree_get_idx(0, state_seq),
                                    backwd_variables_seq)

        return jnp.mean(mc_samples), 0.0
    
def check_linear_gaussian_elbo(p: LinearGaussianHMM, num_seqs, seq_length):
    key_params, key_gen = jax.random.split(jax.random.PRNGKey(0), 2)
    theta = p.get_random_params(key_params)

    seqs = p.sample_multiple_sequences(key_gen, theta, num_seqs, seq_length)[1]

    elbo = LinearGaussianELBO(p, p)

    evidence_reference = vmap(lambda seq: p.likelihood_seq(
        seq, theta))(seqs)
    theta = p.format_params(theta)
    evidence_elbo = vmap(lambda seq: elbo(seq, len(seq)-1, theta, theta)[0])(seqs)

    print('ELBO sanity check:', jnp.mean(
        jnp.abs(evidence_elbo - evidence_reference)))


def check_general_elbo(mc_key, p: LinearGaussianHMM, num_seqs, seq_length, num_samples):

    key_params, key_gen = jax.random.split(jax.random.PRNGKey(0), 2)
    theta = p.get_random_params(key_params)

    seqs = p.sample_multiple_sequences(key_gen, theta, num_seqs, seq_length)[1]
    mc_keys = jax.random.split(mc_key, num_seqs)
    elbo = GeneralBackwardELBO(p, p, num_samples)


    theta = p.format_params(theta)
    reference_elbo = vmap(lambda seq:LinearGaussianELBO(p,p)(seq, len(seq)-1, theta, theta)[0])(seqs)

    evidence_elbo = vmap(lambda key, seq: elbo(
        key, seq, len(seq) - 1, theta, theta)[0])(mc_keys, seqs)
    print('ELBO sanity check:', jnp.mean(
        jnp.abs(evidence_elbo - reference_elbo)))


# if '__name__' == '__main__':
#     check_general_elbo(jax.random.PRNGKey(0))