import jax 
from jax import numpy as jnp
from jax import config as config
import seaborn as sns 
from backward_ica.stats.hmm import LinearGaussianHMM
from backward_ica.utils import * 
from datetime import datetime 
import os 
from backward_ica.offline_smoothing import LinearGaussianELBO, OfflineVariationalAdditiveSmoothing
from backward_ica.online_smoothing import OnlineVariationalAdditiveSmoothing, init_standard, update_IS, update_PaRIS

import backward_ica.stats.hmm as hmm
import backward_ica.stats as stats
import backward_ica.variational as variational

from backward_ica.stats.kalman import Kalman
import pandas as pd 
import matplotlib.pyplot as plt
import math

# config.update('jax_disable_jit',True)

enable_x64(True)

date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
output_dir = os.path.join('experiments','online', date)
os.makedirs(output_dir, exist_ok=True)


parser = argparse.ArgumentParser()
parser.set_defaults(seed=0, 
                    load_p_from='',
                    load_q_from='',
                    functional='x1x2',
                    at_epoch=None,
                    num_replicas=100,
                    seq_length=50,
                    state_dim=2, 
                    obs_dim=5,
                    transition_matrix_conditionning='diagonal',
                    range_transition_map_params=(0,1),
                    default_prior_base_scale = math.sqrt(1e-2),
                    default_transition_base_scale = math.sqrt(1e-2),
                    default_emission_base_scale = math.sqrt(1e-2),
                    transition_bias=True,
                    emission_bias=True,
                    num_samples=100)

args = parser.parse_args()

save_args(args, 'args', os.path.join(output_dir))



def get_additive_functionals(p, q:LinearGaussianHMM, theta, phi, functional_name='elbo'):

    if functional_name == 'elbo':
        offline_functional = offline_elbo_functional(p, q)
        online_functional = online_elbo_functional(p, q)
        oracle = lambda obs_seq: LinearGaussianELBO(p,q)(obs_seq, len(obs_seq)-1, p.format_params(theta), q.format_params(phi))

    elif functional_name == 'x':
        offline_functional = state_smoothing_functional(p, q)
        online_functional = state_smoothing_functional(p, q)
        oracle = lambda obs_seq: jnp.sum(Kalman.smooth_seq(obs_seq, q.format_params(phi))[0], axis=0) / len(obs_seq)

    elif functional_name == 'x1x2':
        offline_functional = offline_x1_x2_functional(p, q)
        online_functional = online_x1_x2_functional(p, q)
        oracle = lambda obs_seq: jnp.prod(q.smooth_seq(obs_seq, phi, lag=1).mean, axis=1) / len(obs_seq)
    else: 
        raise NotImplementedError

    return online_functional, offline_functional, lambda obs_seq: jnp.linalg.norm(oracle(obs_seq), ord=1)

def get_models(args, key_theta, key_phi):

    if args.load_p_from != '':
        p_args = load_args('args', args.load_p_from)
        stats.set_parametrization(p_args)

        args.state_dim = p_args.state_dim 
        args.obs_dim = p_args.obs_dim 
        
        p = hmm.get_generative_model(p_args)
        theta = load_params('theta_star', args.load_p_from)

    else: 
        p = LinearGaussianHMM(args.state_dim, 
                            args.obs_dim, 
                            args.transition_matrix_conditionning, 
                            args.range_transition_map_params, 
                            args.transition_bias, 
                            args.emission_bias)

        theta = p.get_random_params(key_theta, args)


    if args.load_q_from != '':
        q_args = load_args('args', args.load_q_from)
        q_args.state_dim, q_args.obs_dim = args.state_dim, args.obs_dim 
        if q_args.model.split('_')[0] == 'linear': 
            q_args.model = 'linear'
        q = variational.get_variational_model(q_args, 
                                            p=p)
        phi = load_params('phi', args.load_q_from)

    else: 
            
        q = LinearGaussianHMM(args.state_dim, 
                            args.obs_dim, 
                            args.transition_matrix_conditionning, 
                            args.range_transition_map_params, 
                            args.transition_bias, 
                            args.emission_bias)
        phi = q.get_random_params(key_phi, args)
    if args.at_epoch is not None: 
        phi = phi[args.at_epoch]

    return p,theta,q,phi

def get_offline_estimator(theta, phi, additive_functional):

    offline_estimator = lambda key, obs_seq: jnp.linalg.norm(OfflineVariationalAdditiveSmoothing(p, q, additive_functional, args.num_samples)(key, 
                                                                                    obs_seq, 
                                                                                    len(obs_seq)-1, 
                                                                                    p.format_params(theta), 
                                                                                    q.format_params(phi)), ord=1)

    return jax.vmap(offline_estimator, in_axes=(0,None))

def get_online_estimator(theta, phi, additive_functional, version):
    
    if version == 'IS':

        online_elbo = partial(OnlineVariationalAdditiveSmoothing(p=p, 
                                                                q=q, 
                                                                init_func=init_standard, 
                                                                update_func=update_IS, 
                                                                additive_functional=additive_functional,
                                                                num_samples=args.num_samples,
                                                                normalizer=None).batch_compute,
                            theta=p.format_params(theta),
                            phi=q.format_params(phi))
    
    elif version == 'normalized IS':

        online_elbo = partial(OnlineVariationalAdditiveSmoothing(p=p, 
                                                                q=q, 
                                                                init_func=init_standard, 
                                                                update_func=update_IS, 
                                                                additive_functional=additive_functional,
                                                                num_samples=args.num_samples,
                                                                normalizer=exp_and_normalize).batch_compute,
                            theta=p.format_params(theta),
                            phi=q.format_params(phi))

    elif version == 'PaRIS':
        num_samples = int(jnp.sqrt(args.num_samples ** 3) / 2)
        
        online_elbo = partial(OnlineVariationalAdditiveSmoothing(p=p, 
                                                                q=q, 
                                                                init_func=init_standard, 
                                                                update_func=partial(update_PaRIS, num_paris_samples=2), 
                                                                additive_functional=additive_functional,
                                                                num_samples=num_samples,
                                                                normalizer=exp_and_normalize).batch_compute,
                            theta=p.format_params(theta),
                            phi=q.format_params(phi))

    return jax.vmap(lambda key, obs_seq: jnp.linalg.norm(online_elbo(key, obs_seq), ord=1), in_axes=(0,None))



key = jax.random.PRNGKey(args.seed)

key, key_theta, key_phi = jax.random.split(key, 3)

p, theta, q, phi = get_models(args, key_theta, key_phi)

key, key_seq = jax.random.split(key, 2)
_ , obs_seq = p.sample_seq(key_seq, theta, args.seq_length)


online_additive_functional, offline_additive_functional, oracle = get_additive_functionals(p, q, theta, phi, args.functional)

true_value = oracle(obs_seq)


key, *keys = jax.random.split(key, args.num_replicas + 1)
keys = jnp.array(keys)

offline_values = get_offline_estimator(theta, phi, offline_additive_functional)(keys, obs_seq)


online_IS_values = get_online_estimator(theta, phi, online_additive_functional, 'IS')(keys, obs_seq)
online_normalized_IS_values = get_online_estimator(theta, phi, online_additive_functional, 'normalized IS')(keys, obs_seq)
online_PaRIS_values = get_online_estimator(theta, phi, online_additive_functional, 'PaRIS')(keys, obs_seq)


offline_errors = pd.DataFrame((offline_values - true_value) / jnp.abs(true_value))
online_IS_errors = pd.DataFrame((online_IS_values - true_value) / jnp.abs(true_value))
online_normalized_IS_errors = pd.DataFrame((online_normalized_IS_values - true_value) / jnp.abs(true_value))
online_PaRIS_errors = pd.DataFrame((online_PaRIS_values - true_value) / jnp.abs(true_value))


errors = pd.concat([offline_errors, online_IS_errors, online_normalized_IS_errors, online_PaRIS_errors], axis=1)

errors.columns = ['Offline', 'Online IS', 'Online normalized IS', 'Online PaRIS']

errors.plot(kind='box')
plt.xlabel('Method')
plt.ylabel('Relative error to true functional')

plt.suptitle(f'N={args.num_samples}\nT={args.seq_length}\nd_x={args.state_dim}\nd_y={args.obs_dim}\nfunctional:{args.functional}')
plt.savefig(os.path.join(output_dir, 'errors'))

