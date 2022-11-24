import jax 
from jax import numpy as jnp
import seaborn as sns 
from backward_ica.stats.hmm import LinearGaussianHMM
from backward_ica.utils import * 
from datetime import datetime 
import os 
from backward_ica.offline_elbos import GeneralBackwardELBO, LinearGaussianELBO
from backward_ica.online_smoothing import OnlinePaRISAdditiveSmoothing, OnlineISAdditiveSmoothing

import backward_ica.stats.hmm as hmm
import backward_ica.stats as stats
import backward_ica.variational as variational


import pandas as pd 
import matplotlib.pyplot as plt
import math

enable_x64(True)

date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
output_dir = os.path.join('experiments','online', date)
os.makedirs(output_dir, exist_ok=True)


parser = argparse.ArgumentParser()
parser.set_defaults(seed=0, 
                    load_p_from='',
                    load_q_from='',
                    at_epoch=None,
                    num_replicas=100,
                    seq_length=50,
                    state_dim=2, 
                    obs_dim=2,
                    transition_matrix_conditionning='diagonal',
                    range_transition_map_params=(0,1),
                    default_prior_base_scale = math.sqrt(1e-2),
                    default_transition_base_scale = math.sqrt(1e-2),
                    default_emission_base_scale = math.sqrt(1e-2),
                    transition_bias=True,
                    emission_bias=True,
                    num_samples=20)

args = parser.parse_args()

save_args(args, 'args', os.path.join(output_dir))


key = jax.random.PRNGKey(args.seed)

key, key_theta, key_phi = jax.random.split(key, 3)


def get_models(args):

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

p, theta, q, phi = get_models(args)

key, key_seq = jax.random.split(key, 2)
state_seq, obs_seq = p.sample_seq(key_seq, theta, args.seq_length)


def get_offline_estimator(theta, phi):

    offline_elbo = lambda key, obs_seq: GeneralBackwardELBO(p, q, args.num_samples)(key, 
                                                                                    obs_seq, 
                                                                                    len(obs_seq)-1, 
                                                                                    p.format_params(theta), 
                                                                                    q.format_params(phi))

    return jax.vmap(offline_elbo, in_axes=(0,None))

def get_online_estimator(theta, phi, functionals, version):
    
    if version == 'IS':

        online_elbo = partial(OnlineISAdditiveSmoothing(p, q, functionals, None, args.num_samples).batch_compute,
                            theta=p.format_params(theta),
                            phi=q.format_params(phi))
    
    elif version == 'normalized IS':

        online_elbo = partial(OnlineISAdditiveSmoothing(p, q, functionals, exp_and_normalize, args.num_samples).batch_compute,
                            theta=p.format_params(theta),
                            phi=q.format_params(phi))
                            

    elif version == 'PaRIS':
        num_samples = int(jnp.sqrt(args.num_samples ** 3) / 2)
        
        online_elbo = partial(OnlinePaRISAdditiveSmoothing(p, q, functionals, exp_and_normalize, num_samples).batch_compute,
                            theta=p.format_params(theta),
                            phi=q.format_params(phi))
    return jax.vmap(online_elbo, in_axes=(0,None))


key, *keys = jax.random.split(key, args.num_replicas + 1)
keys = jnp.array(keys)

true_elbo = LinearGaussianELBO(p,q)(obs_seq, len(obs_seq)-1, p.format_params(theta), q.format_params(phi))


functionals = elbo_forward_functionals
offline_values = get_offline_estimator(theta, phi)(keys, obs_seq)

online_IS_values = get_online_estimator(theta, phi, functionals, 'IS')(keys, obs_seq)
online_normalized_IS_values = get_online_estimator(theta, phi, functionals, 'normalized IS')(keys, obs_seq)
online_PaRIS_values = get_online_estimator(theta, phi, functionals, 'PaRIS')(keys, obs_seq)


offline_errors = pd.DataFrame((offline_values - true_elbo) / jnp.abs(true_elbo))
online_IS_errors = pd.DataFrame((online_IS_values - true_elbo) / jnp.abs(true_elbo))
online_normalized_IS_errors = pd.DataFrame((online_normalized_IS_values - true_elbo) / jnp.abs(true_elbo))
online_PaRIS_errors = pd.DataFrame((online_PaRIS_values - true_elbo) / jnp.abs(true_elbo))


errors = pd.concat([offline_errors, online_IS_errors, online_normalized_IS_errors, online_PaRIS_errors], axis=1)

errors.columns = ['Offline', 'Online IS', 'Online normalized IS', 'Online PaRIS']

errors.plot(kind='box')
plt.xlabel('Method')
plt.ylabel('Relative error to true ELBO')

plt.suptitle(f'N={args.num_samples}\nT={args.seq_length}\nd_x={args.state_dim}\nd_y={args.obs_dim}')
plt.savefig(os.path.join(output_dir, 'errors'))

