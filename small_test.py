import jax, jax.lax as lax, jax.numpy as jnp, jax.tree_util as tree_util
from jax.experimental.maps import xmap
import time
from backward_ica.stats.hmm import LinearGaussianHMM
from backward_ica.utils import *
import operator
from functools import reduce
# jax.config.update('jax_disable_jit', True)




# def f(a, b, **kwargs):

#     return a * b



# a = jnp.arange(0,100)
# b = 1.2
# other = {'c': 50, 'd':jnp.zeros((20,))}
# print(named_vmap(f, {'a'}))

def f(dict_input):
    return dict_input['a0']['a1'] * dict_input['a0']['a2']

test = {'a0':{'a1':jnp.arange(0,10), 
            'a2':2.3 * jnp.ones((100,))}}

print(named_vmap(f, {'a0':{'a1':0}}, test))

# print(test['a0']['a1'])

# def f(x,y):
#     return x*y

# x = jnp.ones((100,))
# y = jnp.ones((50,))


# test = xmap(f, in_axes={0:'x', 1:'y'}, out_axes=['z', ...])(x,y)



# key = jax.random.PRNGKey(0)


# p = LinearGaussianHMM(2, 2, 'diagonal', (0,1), True, True)
# q = LinearGaussianHMM(2, 2, 'diagonal', (0,1), True, True)



# key, key_theta, key_phi = jax.random.split(key, 3)

# theta = p.get_random_params(key_theta)
# phi = q.get_random_params(key_phi)

# key, key_seq = jax.random.split(key, 2)
# seq_length = 100
# state_seq, obs_seq = p.sample_seq(key_seq, theta, seq_length)

# # state_seq = q.compute_state_seq(obs_seq, len(obs_seq)-1, q.format_params(phi))

# num_samples = 100


# phi = q.format_params(phi)
# phi.compute_covs()

# def samples_and_log_probs(key, q_params):
#     samples = jax.vmap(q.filt_dist.sample, in_axes=(0, None))(jax.random.split(key, num_samples), q_params)
#     log_probs = jax.vmap(q.filt_dist.logpdf, in_axes=(0, None))(samples, q_params)
#     return samples, log_probs


# def step_store(carry, x):

#     prev_state, prev_log_probs = carry
#     key, obs = x

#     state = q.new_state(obs, prev_state, phi)

#     samples, log_probs = samples_and_log_probs(key, q.filt_params_from_state(state, phi))

#     return (state, log_probs), prev_log_probs + log_probs


# def step_recompute(carry, x):

#     prev_state, prev_samples, prev_log_probs = carry
#     key, obs = x

#     state = q.new_state(obs, prev_state, phi)

#     samples, log_probs = samples_and_log_probs(key, q.filt_params_from_state(state, phi))

#     prev_log_probs = jax.vmap(q.filt_dist.logpdf, in_axes=(0, None))(prev_samples, q.filt_params_from_state(prev_state, phi))

#     return (state, samples, log_probs), prev_log_probs + log_probs

# init_state = q.init_state(obs_seq[0], phi)

# key, init_key = jax.random.split(key, 2)
# samples, log_probs = samples_and_log_probs(init_key, q.filt_params_from_state(init_state, phi))


# key, *key_seq = jax.random.split(key, seq_length)


# result_store = jnp.mean(lax.scan(step_store, 
#                 init=(init_state, log_probs), 
#                 xs=(jnp.array(key_seq), obs_seq[1:]))[1])


# result_recompute = jnp.mean(lax.scan(step_recompute, 
#                 init=(init_state, samples, log_probs), 
#                 xs=(jnp.array(key_seq), obs_seq[1:]))[1])


# for _ in range(20):
        
#     time0 = time.time()
#     print(result_store.block_until_ready())
#     print('Time when storing:',time.time() - time0)

#     time0 = time.time()
#     print(result_recompute.block_until_ready())
#     print('Time when recomputing:', time.time() - time0)


#     print('-----')