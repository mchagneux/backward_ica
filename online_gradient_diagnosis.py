from backward_ica.stats.hmm import LinearGaussianHMM
from backward_ica.online_smoothing import OnlineNormalizedISELBO, OnlineNormalizedISELBOPrecompute, OnlineProposalResampling
from backward_ica.offline_smoothing import LinearGaussianELBO, GeneralBackwardELBO
import jax, jax.numpy as jnp
from functools import partial
from backward_ica.utils import enable_x64
import os 
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree as ravel
d_x = 5
d_y = 5
import pandas as pd 
import matplotlib.pyplot as plt 

enable_x64(True)

jax.config.update('jax_disable_jit', False)

p = LinearGaussianHMM(
                    state_dim=d_x,
                    obs_dim=d_y, 
                    transition_matrix_conditionning='diagonal',
                    range_transition_map_params=(-0.95,0.95),
                    transition_bias=False,
                    emission_bias=False)

q = LinearGaussianHMM(state_dim=d_x,
                    obs_dim=d_y, 
                    transition_matrix_conditionning='diagonal',
                    range_transition_map_params=(-0.95,0.95),
                    transition_bias=False,
                    emission_bias=False)



num_samples_oracle = 10000
num_samples = 100
num_replicas = 100
seq_length = 50
num_runs = 3
compute_grads = True
online_methods = True
name_method_2 = 'proposal'

key = jax.random.PRNGKey(5)
key, key_theta = jax.random.split(key, 2)
theta = p.get_random_params(key_theta)

path = 'experiments/online/compare_naive_and_proposal_no_correction'

os.makedirs(path, exist_ok=True)

def scalar_relative_error(oracle, estimate):
    return (oracle - estimate) / jnp.abs(oracle)

def cosine_similarity(oracle, estimate):
    return (oracle @ estimate) / (jnp.linalg.norm(oracle, ord=2) * jnp.linalg.norm(estimate, ord=2))

if isinstance(p, LinearGaussianHMM) and isinstance(q, LinearGaussianHMM):
    oracle = LinearGaussianELBO(p, q)
else: 
    oracle = GeneralBackwardELBO(p, q, 100000)

offline_elbo = GeneralBackwardELBO(p, q, num_samples)

online_elbo = OnlineNormalizedISELBO(p, q, num_samples)
online_elbo_2 = OnlineProposalResampling(p, q, num_samples)

if not isinstance(oracle, LinearGaussianELBO):
    oracle_elbo = lambda obs_seq, compute_up_to, theta, phi: oracle(
                                                    jax.random.PRNGKey(0), 
                                                    obs_seq,
                                                    compute_up_to, 
                                                    theta, 
                                                    phi)
    

else: 
    oracle_elbo = oracle

def oracle_elbo_and_grad_func(obs_seq, theta, phi):

    if compute_grads: 

        elbo, grad_elbo = jax.value_and_grad(lambda obs_seq, phi: oracle_elbo(obs_seq, 
                                                    len(obs_seq)-1, 
                                                    p.format_params(theta), 
                                                    q.format_params(phi))[0], 
                                                    argnums=1)(obs_seq, phi)
        return len(obs_seq)*elbo, ravel(grad_elbo)[0]

    else: 
        elbo = oracle_elbo(obs_seq, 
                    len(obs_seq)-1, 
                    p.format_params(theta), 
                    q.format_params(phi))[0]
        return len(obs_seq)*elbo, jnp.zeros((2,))


jit_oracle_elbo_and_grad_func = jax.jit(oracle_elbo_and_grad_func)


def offline_elbo_and_grad_func(key, obs_seq, theta, phi):
    if compute_grads: 

        elbo, grad_elbo = jax.value_and_grad(lambda obs_seq, phi: offline_elbo(key, obs_seq, 
                                                    len(obs_seq)-1, 
                                                    p.format_params(theta), 
                                                    q.format_params(phi))[0], 
                                                    argnums=1)(obs_seq, phi)
        return len(obs_seq)*elbo, ravel(grad_elbo)[0]


    else: 
        elbo = offline_elbo(key, obs_seq, 
                            len(obs_seq)-1, 
                            p.format_params(theta), 
                            q.format_params(phi))[0]
        return len(obs_seq)*elbo, jnp.zeros((2,))

jit_offline_elbo_and_grad_func = jax.jit(jax.vmap(offline_elbo_and_grad_func, in_axes=(0,None, None, None)))


def online_elbo_and_grad_func(key, obs_seq, theta, phi):
    if compute_grads: 
        (elbo, weights), grad_elbo = jax.value_and_grad(lambda obs_seq, phi: online_elbo.batch_compute(key, obs_seq, 
                                                                p.format_params(theta), 
                                                                phi), 
                                                                argnums=1, has_aux=True)(obs_seq, phi)


        return (len(obs_seq)*elbo, ravel(grad_elbo)[0]), weights
    else: 
        elbo, weights = online_elbo.batch_compute(key, obs_seq, 
                                        p.format_params(theta), 
                                        phi)
        return (len(obs_seq)*elbo, weights), jnp.zeros((2,))

def online_elbo_and_grad_func_2(key, obs_seq, theta, phi):
    if compute_grads: 
        (elbo, weights), grad_elbo = jax.value_and_grad(lambda obs_seq, phi: online_elbo_2.batch_compute(key, obs_seq, 
                                                                p.format_params(theta), 
                                                                phi), 
                                                                argnums=1, has_aux=True)(obs_seq, phi)


        return (len(obs_seq)*elbo, ravel(grad_elbo)[0]), weights
    else: 
        elbo, weights = online_elbo_2.batch_compute(key, obs_seq, 
                                        p.format_params(theta), 
                                        phi)
        return (len(obs_seq)*elbo, weights), jnp.zeros((2,))
    

jit_online_elbo_and_grad_func = jax.jit(jax.vmap(online_elbo_and_grad_func, in_axes=(0,None, None, None)))
jit_online_elbo_and_grad_func_2 = jax.jit(jax.vmap(online_elbo_and_grad_func_2, in_axes=(0,None, None, None)))


def experiment(key, exp_id, exp_name):



    key, key_phi = jax.random.split(key, 2)

    phi = q.get_random_params(key_phi)

    key, key_seq = jax.random.split(key, 2)

    state_seq, obs_seq = p.sample_seq(key_seq, theta, seq_length)

    print('Computing oracle results...')
    oracle_elbo, oracle_grad_elbo = jit_oracle_elbo_and_grad_func(obs_seq, theta, phi)
    
    print('Computing offline results...')
    offline_elbos, offline_grads_elbo = jit_offline_elbo_and_grad_func(jax.random.split(key, num_replicas), obs_seq, theta, phi)
    offline_elbo_errors = jax.vmap(scalar_relative_error, in_axes=(None,0))(oracle_elbo, offline_elbos)
    offline_grad_elbo_errors = jax.vmap(cosine_similarity, in_axes=(None, 0))(oracle_grad_elbo, offline_grads_elbo)

    if online_methods: 

        print('Computing recursive results method 1...')
        (online_elbos, online_grads_elbo), weights = jit_online_elbo_and_grad_func(jax.random.split(key, num_replicas), obs_seq, theta, phi)
        online_elbo_errors = jax.vmap(scalar_relative_error, in_axes=(None,0))(oracle_elbo, online_elbos)
        if compute_grads: online_grad_elbo_errors = jax.vmap(cosine_similarity, in_axes=(None, 0))(oracle_grad_elbo, online_grads_elbo)
        # jnp.save(os.path.join(path, 'weights_method_1.npy'), weights)

        print('Computing recursive results method 2...')

        (online_elbos_2, online_grads_elbo_2), weights = jit_online_elbo_and_grad_func_2(jax.random.split(key, num_replicas), obs_seq, theta, phi)
        online_elbo_errors_2 = jax.vmap(scalar_relative_error, in_axes=(None,0))(oracle_elbo, online_elbos_2)
        # jnp.save(os.path.join(path, 'weights_method_2.npy'), weights)

        if compute_grads: 
            online_grad_elbo_errors_2 = jax.vmap(cosine_similarity, in_axes=(None, 0))(oracle_grad_elbo, online_grads_elbo_2)

        elbo_errors = pd.DataFrame(jnp.array([offline_elbo_errors, online_elbo_errors, online_elbo_errors_2]).T, columns=['Offline',  'Online Naive SNIS', f'Online {exp_name}'])
        if compute_grads: 
            grad_elbo_errors = pd.DataFrame(jnp.array([offline_grad_elbo_errors, online_grad_elbo_errors, online_grad_elbo_errors_2]).T, columns=['Offline',  'Online Naive SNIS', f'Online {exp_name}'])


    else: 
        elbo_errors = pd.DataFrame(offline_elbo_errors, columns=['Offline'])
        if compute_grads:
            grad_elbo_errors = pd.DataFrame(offline_grad_elbo_errors, columns=['Offline'])


    elbo_errors.plot(kind='box')
    plt.savefig(os.path.join(path, f'elbo_errors_{exp_id}'))

    if compute_grads: 
        grad_elbo_errors.plot(kind='box')
        plt.savefig(os.path.join(path, f'grad_elbo_errors_{exp_id}'))

    print('---')

for exp_id in range(num_runs):
    key, subkey = jax.random.split(key, 2)
    experiment(subkey, exp_id, name_method_2)


              





    









