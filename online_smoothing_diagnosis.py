# %%
import argparse
import haiku as hk
import jax
import jax.numpy as jnp
import backward_ica.stats.hmm as hmm
import backward_ica.utils as utils
import backward_ica.smc as smc
from backward_ica.offline_elbos import GeneralBackwardELBO, LinearGaussianELBO, OnlineGeneralBackwardELBO, OnlineGeneralBackwardELBOSpecialInit, OnlineGeneralBackwardELBOV2
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import random
import pandas as pd
import math
gaussian_distance = hmm.Gaussian.KL

normalizer = smc.exp_and_normalize
# normalizer = lambda x: jnp.mean(jnp.exp(x))


def compute_distances_filt_and_backwd(q: hmm.BackwardSmoother, samples_seq, filt_params_seq, backwd_params_seq):
    samples_seq = utils.tree_dropfirst(samples_seq)
    filt_params_seq = utils.tree_droplast(filt_params_seq)

    def compute_distance_at_t(x_tp1, q_t_params, q_t_tp1_params):
        q_t_tp1_xi_tp1_params = q.backwd_kernel.map(x_tp1, q_t_tp1_params)
        return gaussian_distance(q_t_tp1_xi_tp1_params, q_t_params.out)
    return jax.vmap(jax.vmap(compute_distance_at_t, in_axes=(0, None, None)))(samples_seq, filt_params_seq, backwd_params_seq)


def get_sampler_and_elbos(key, args, args_p, args_q):

    key, *subkeys = jax.random.split(key, 3)
    p, theta_star = utils.get_generative_model(args_p, subkeys[0])
    q, phi = utils.get_variational_model(args_q, p, subkeys[1])

    if args.trained_model:
        theta_star = utils.load_params(
            'theta', os.path.join(args.exp_dir, args.method_name))
        phi = utils.load_params('phi', args.method_dir)[1]

    if args_p.p_version == 'linear':
        closed_form_elbo = jax.jit(jax.vmap(lambda obs_seq: LinearGaussianELBO(p, q)(obs_seq,
                                                                                     p.format_params(
                                                                                         theta_star),
                                                                                     q.format_params(phi))))
    else:
        def closed_form_elbo(obs_seq): return 0

    keys = jax.random.split(key, args.num_seqs)

    offline_mc_elbo = jax.vmap(in_axes=(0, 0, None), fun=lambda key, obs_seq, num_samples: GeneralBackwardELBO(p, q, num_samples)(
        key,
        obs_seq,
        p.format_params(theta_star),
        q.format_params(phi)))

    online_mc_elbo = jax.vmap(in_axes=(0, 0, None), fun=lambda key, obs_seq, num_samples: OnlineGeneralBackwardELBO(p, q, normalizer, num_samples)(
        key,
        obs_seq,
        p.format_params(theta_star),
        q.format_params(phi)))

    def mc_elbos(obs_seqs, num_samples):
        offline_value = offline_mc_elbo(keys, obs_seqs, num_samples)
        online_value, (samples_seqs, weights_seqs, filt_params_seqs,
                       backwd_params_seqs) = online_mc_elbo(keys, obs_seqs, num_samples)
        return offline_value / len(obs_seqs[0]), online_value / len(obs_seqs[0]), (samples_seqs, weights_seqs, filt_params_seqs, backwd_params_seqs)

    def true_elbo(obs_seqs):
        return closed_form_elbo(obs_seqs) / len(obs_seqs[0])

    def sampler(key):
        return p.sample_multiple_sequences(key, theta_star, args.num_seqs, args.seq_length)

    return sampler, mc_elbos, true_elbo


def get_offline_elbo(p, theta, q, phi):
    return lambda keys, obs_seqs, num_samples: jax.vmap(in_axes=(0, 0, None), fun=lambda key, obs_seq, num_samples: GeneralBackwardELBO(p, q, num_samples)(
        key,
        obs_seq,
        p.format_params(theta),
        q.format_params(phi)))(keys, obs_seqs, num_samples)


def get_sampler_and_true_elbo(key, args, args_p, args_q):

    subkeys = jax.random.split(key, 2)
    p, theta_star = utils.get_generative_model(args_p, subkeys[0])
    q, phi = utils.get_variational_model(args_q, p, subkeys[1])

    if args.trained_model:
        theta_star = utils.load_params(
            'theta', os.path.join(args.exp_dir, args.method_name))
        phi = utils.load_params('phi', args.method_dir)[1]

    if args_p.p_version == 'linear':
        closed_form_elbo = jax.jit(jax.vmap(lambda obs_seq: LinearGaussianELBO(p, q)(obs_seq,
                                                                                     p.format_params(
                                                                                         theta_star),
                                                                                     q.format_params(phi))))
    else:
        def closed_form_elbo(obs_seq): return 0

    def true_elbo(obs_seqs):
        return closed_form_elbo(obs_seqs)

    def sampler(key):
        return p.sample_multiple_sequences(key, theta_star, args.num_seqs, args.seq_length)

    return p, q, theta_star, phi, sampler, true_elbo


def get_online_elbo(p, theta, q, phi, version):

    if version == 0:
        online_mc_elbo = jax.vmap(in_axes=(0, 0, None), fun=lambda key, obs_seq, num_samples: OnlineGeneralBackwardELBO(p, q, normalizer, num_samples)(
            key,
            obs_seq,
            p.format_params(theta),
            q.format_params(phi)))
    elif version == 1:
        online_mc_elbo = jax.vmap(in_axes=(0, 0, None), fun=lambda key, obs_seq, num_samples: OnlineGeneralBackwardELBOV2(p, q, normalizer, num_samples)(
            key,
            obs_seq,
            p.format_params(theta),
            q.format_params(phi)))

    def online_elbo(keys, obs_seqs, num_samples):
        elbo_values, aux = online_mc_elbo(keys, obs_seqs, num_samples)
        return elbo_values, aux

    return online_elbo


def main(args, args_p, args_q):

    key = jax.random.PRNGKey(0)

    utils.set_parametrization(args_p)

    if args.evolution_wrt_seq_length != -1:

        key, *subkeys = jax.random.split(key, 3)

        sampler, mc_elbos, true_elbo = get_sampler_and_elbos(
            subkeys[0], args, args_p, args_q)

        mc_elbos_jit = jax.jit(
            lambda obs_seqs: mc_elbos(obs_seqs, args.num_samples))

        state_seqs, obs_seqs = sampler(subkeys[1])

        key, *subkeys = jax.random.split(key, args.num_seqs)

        true_values_list = dict()
        offline_values_list = dict()
        online_values_list = dict()

        for stopping_point in range(args.evolution_wrt_seq_length, args.seq_length, args.evolution_wrt_seq_length):

            true_values = true_elbo(obs_seqs[:, :stopping_point])
            offline_values, online_values = mc_elbos_jit(
                obs_seqs[:, :stopping_point])[:-1]

            true_values_list[stopping_point] = true_values
            offline_values_list[stopping_point] = offline_values
            online_values_list[stopping_point] = online_values

        true_values = true_elbo(obs_seqs)
        offline_values, online_values, (samples_seqs, weights_seqs,
                                        filt_params_seqs, backwd_params_seqs) = mc_elbos_jit(obs_seqs)

        true_values_list[args.seq_length] = true_values
        offline_values_list[args.seq_length] = offline_values
        online_values_list[args.seq_length] = online_values

        errors_online = dict()
        errors_offline = dict()

        for k in true_values_list.keys():
            errors_online_k = jnp.abs(
                true_values_list[k] - online_values_list[k])
            errors_online[k] = {k: v.tolist()
                                for k, v in enumerate(errors_online_k)}
            errors_offline_k = jnp.abs(
                true_values_list[k] - offline_values_list[k])
            errors_offline[k] = {k: v.tolist()
                                 for k, v in enumerate(errors_offline_k)}

        errors_online = pd.DataFrame(errors_online).apply(
            pd.Series).unstack().reset_index()
        errors_online['Method'] = 'Online'
        errors_offline = pd.DataFrame(errors_offline).apply(
            pd.Series).unstack().reset_index()
        errors_offline['Method'] = 'Offline'
        errors = pd.concat([errors_online, errors_offline],
                           axis=0).reset_index().drop(columns='index')

        errors.columns = ['Seq length', 'Seq nb', 'Value', 'Method']
        errors.to_csv(os.path.join(args.save_dir, 'errors_wrt_length.csv'))

        sns.violinplot(data=errors,
                       x='Seq length',
                       y='Value',
                       hue='Method',
                       inner='point')

        plt.savefig(os.path.join(args.save_dir,
                    'errors_wrt_length_violinplot'))
        plt.close()

    if args.evolution_wrt_num_samples != -1:

        key, *subkeys = jax.random.split(key, 3)

        sampler, mc_elbos, true_elbo = get_sampler_and_elbos(
            subkeys[0], args, args_p, args_q)

        state_seqs, obs_seqs = sampler(subkeys[1])

        offline_values_list = dict()
        online_values_list = dict()

        true_values = true_elbo(obs_seqs)

        for n_samples in range(1, args.num_samples, args.evolution_wrt_num_samples):

            offline_values, online_values = mc_elbos(obs_seqs, n_samples)[:-1]

            offline_values_list[n_samples] = offline_values
            online_values_list[n_samples] = online_values

        offline_values, online_values, (samples_seqs, weights_seqs, filt_params_seqs,
                                        backwd_params_seqs) = mc_elbos(obs_seqs, args.num_samples)

        offline_values_list[args.num_samples] = offline_values
        online_values_list[args.num_samples] = online_values

        errors_online = dict()
        errors_offline = dict()

        for k in offline_values_list.keys():
            errors_online_k = jnp.abs(true_values - online_values_list[k])
            errors_online[k] = {k: v.tolist()
                                for k, v in enumerate(errors_online_k)}
            errors_offline_k = jnp.abs(true_values - offline_values_list[k])
            errors_offline[k] = {k: v.tolist()
                                 for k, v in enumerate(errors_offline_k)}

        errors_online = pd.DataFrame(errors_online).apply(
            pd.Series).unstack().reset_index()
        errors_online['Method'] = 'Online'
        errors_offline = pd.DataFrame(errors_offline).apply(
            pd.Series).unstack().reset_index()
        errors_offline['Method'] = 'Offline'
        errors = pd.concat([errors_online, errors_offline],
                           axis=0).reset_index().drop(columns='index')

        errors.columns = ['Num samples', 'Seq nb', 'Value', 'Method']
        errors.to_csv(os.path.join(
            args.save_dir, 'errors_wrt_num_samples.csv'))
        sns.violinplot(data=errors,
                       x='Num samples',
                       y='Value',
                       hue='Method',
                       inner='point')

        plt.savefig(os.path.join(args.save_dir,
                    'errors_wrt_num_samples_violinplot'))
        plt.close()

    elif args.evolution_wrt_dim != -1:
        key, *subkeys = jax.random.split(key, 3)

        true_values_list = dict()
        offline_values_list = dict()
        online_values_list = dict()

        for dim in range(args.evolution_wrt_dim, args.state_dim+1, args.evolution_wrt_dim):
            args_p.state_dim = args_q.state_dim = dim
            args_p.obs_dim = args_q.obs_dim = dim

            sampler, mc_elbos, true_elbo = get_sampler_and_elbos(
                subkeys[0], args, args_p, args_q)

            state_seqs, obs_seqs = sampler(subkeys[1])

            true_values = true_elbo(obs_seqs)
            offline_values, online_values = mc_elbos(
                obs_seqs, args.num_samples)[:-1]

            true_values_list[dim] = true_values
            offline_values_list[dim] = offline_values
            online_values_list[dim] = online_values

        true_values = true_elbo(obs_seqs)
        offline_values, online_values, (samples_seqs, weights_seqs, filt_params_seqs,
                                        backwd_params_seqs) = mc_elbos(obs_seqs, args.num_samples)

        true_values_list[args.state_dim] = true_values
        offline_values_list[args.state_dim] = offline_values
        online_values_list[args.state_dim] = online_values

        errors_online = dict()
        errors_offline = dict()

        for k in true_values_list.keys():
            errors_online_k = jnp.abs(
                true_values_list[k] - online_values_list[k])
            errors_online[k] = {k: v.tolist()
                                for k, v in enumerate(errors_online_k)}
            errors_offline_k = jnp.abs(
                true_values_list[k] - offline_values_list[k])
            errors_offline[k] = {k: v.tolist()
                                 for k, v in enumerate(errors_offline_k)}

        errors_online = pd.DataFrame(errors_online).apply(
            pd.Series).unstack().reset_index()
        errors_online['Method'] = 'Online'
        errors_offline = pd.DataFrame(errors_offline).apply(
            pd.Series).unstack().reset_index()
        errors_offline['Method'] = 'Offline'
        errors = pd.concat([errors_online, errors_offline],
                           axis=0).reset_index().drop(columns='index')

        errors.columns = ['State dim', 'Seq nb', 'Value', 'Method']

        errors.to_csv(os.path.join(args.save_dir, 'errors_wrt_state_dim.csv'))

        sns.violinplot(data=errors,
                       x='State dim',
                       y='Value',
                       hue='Method',
                       inner='point')

        # plt.ylabel('$\mathcal{L}(\\theta, \\lambda) - \hat{\mathcal{L}}(\\theta, \\lambda)$')

        plt.savefig(os.path.join(args.save_dir,
                    'errors_wrt_state_dim_violinplot'))
        plt.close()

    elif args.compare_online_versions:
        key, *subkeys = jax.random.split(key, 3)

        p, q, theta, phi, sampler, true_elbo = get_sampler_and_true_elbo(
            key, args, args_p, args_q)

        state_seqs, obs_seqs = sampler(subkeys[0])

        online_mc_elbo_v1 = get_online_elbo(p, theta, q, phi, 0)
        online_mc_elbo_v2 = get_online_elbo(p, theta, q, phi, 1)
        offline_mc_elbo = get_offline_elbo(p, theta, q, phi)

        print('Computing closed-form ELBO...')
        true_values = true_elbo(obs_seqs)
        print('Computing offline Monte Carlo ELBO...')
        offline_values = offline_mc_elbo(jax.random.split(
            subkeys[1], args.num_seqs), obs_seqs, args.num_samples)
        print('Computing naive online Monte Carlo ELBO...')

        online_values_v1, (samples_seqs, weights_seqs, filt_params_seqs, backwd_params_seqs) = online_mc_elbo_v1(
            jax.random.split(subkeys[1], args.num_seqs), obs_seqs, args.num_samples)

        print('Computing online Monte Carlo ELBO v2...')
        online_values_v2, _ = online_mc_elbo_v2(jax.random.split(
            subkeys[1], args.num_seqs), obs_seqs, args.num_samples)

        errors_offline = offline_values - true_values
        errors_online_v1 = online_values_v1 - true_values
        errors_online_v2 = online_values_v2 - true_values

        errors_offline = {k: v.tolist() for k, v in enumerate(errors_offline)}
        errors_online_v1 = {k: v.tolist()
                            for k, v in enumerate(errors_online_v1)}
        errors_online_v2 = {k: v.tolist()
                            for k, v in enumerate(errors_online_v2)}

        errors = pd.DataFrame({'Offline': errors_offline,
                               'Online (v1)': errors_online_v1,
                               'Online (v2)': errors_online_v2})
        errors = errors.unstack().reset_index()
        errors.columns = ['Method', 'Seq nb', 'Value']
        # errors.to_csv(os.path.join(args.save_dir, 'errors.csv'))

        sns.boxplot(data=errors,
                    x='Method',
                    y='Value')

        plt.ylabel(
            '$\hat{\mathcal{L}}(\\theta, \\lambda) - \mathcal{L}(\\theta, \\lambda)$')

        plt.savefig(os.path.join(args.save_dir, 'errors_v1_vs_v2'))
        plt.close()

    elif args.offline_version_only:
        key, *subkeys = jax.random.split(key, 3)
        p, q, theta, phi, sampler, true_elbo = get_sampler_and_true_elbo(
            key, args, args_p, args_q)

        state_seqs, obs_seqs = sampler(subkeys[0])
        offline_mc_elbo = get_offline_elbo(p, theta, q, phi)

        true_elbo_values = true_elbo(obs_seqs)
        offline_values = offline_mc_elbo(jax.random.split(
            subkeys[1], args.num_seqs), obs_seqs, args.num_samples)

        errors = pd.DataFrame(
            (offline_values - true_elbo_values).tolist(), columns=['Errors'])

        sns.boxplot(data=errors)
        plt.savefig(os.path.join(args.save_dir, 'Offline errors'))
        plt.close()

    else:
        key, *subkeys = jax.random.split(key, 3)

        sampler, mc_elbos, true_elbo = get_sampler_and_elbos(
            subkeys[0], args, args_p, args_q)
        state_seqs, obs_seqs = sampler(subkeys[1])

        true_values = true_elbo(obs_seqs)
        offline_values, online_values, (samples_seqs, weights_seqs, filt_params_seqs,
                                        backwd_params_seqs) = mc_elbos(obs_seqs, args.num_samples)

        errors_online = jnp.abs(true_values - online_values)
        errors_offline = jnp.abs(true_values - offline_values)

        errors_offline = {k: v.tolist() for k, v in enumerate(errors_offline)}
        errors_online = {k: v.tolist() for k, v in enumerate(errors_online)}

        # errors_online = pd.DataFrame(errors_online).apply(pd.Series).unstack().reset_index()
        # errors_offline = pd.DataFrame(errors_offline).apply(pd.Series).unstack().reset_index()

        errors = pd.DataFrame(
            {'Online': errors_online, 'Offline': errors_offline})
        errors = errors.unstack().reset_index()
        errors.columns = ['Method', 'Seq nb', 'Value']
        errors.to_csv(os.path.join(args.save_dir, 'errors.csv'))

        sns.boxplot(data=errors,
                    x='Method',
                    y='Value')

        plt.ylabel(
            '$\mathcal{L}(\\theta, \\lambda) - \hat{\mathcal{L}}(\\theta, \\lambda)$')

        plt.savefig(os.path.join(args.save_dir, 'errors'))
        plt.close()

    if not args.offline_version_only:
        print('Computing KL divergences...')
        kl_divergences = jax.vmap(compute_distances_filt_and_backwd, in_axes=(
            None, 0, 0, 0))(q, samples_seqs, filt_params_seqs, backwd_params_seqs)

        seq_nbs, timesteps, sample_nbs = jnp.meshgrid(jnp.arange(kl_divergences.shape[0]),
                                                      jnp.arange(
                                                          kl_divergences.shape[1]),
                                                      jnp.arange(kl_divergences.shape[2]))

        print('Generating boxplots for KL divergences...')

        kl_divergences = jnp.vstack((seq_nbs.ravel(),
                                    timesteps.ravel(),
                                    sample_nbs.ravel(),
                                    kl_divergences.ravel())).T

        kl_divergences = pd.DataFrame({'Seq nb': kl_divergences[:, 0].astype(int).tolist(),
                                       'Timestep': kl_divergences[:, 1].astype(int).tolist(),
                                       'Sample nb': kl_divergences[:, 2].astype(int).tolist(),
                                       'Value': kl_divergences[:, 3].tolist()})

        sns.boxplot(data=kl_divergences, x='Timestep', y='Value')
        plt.tight_layout()
        plt.autoscale(True)
        plt.savefig(os.path.join(args.save_dir, 'kl_divergences'))
        plt.close()

    # seq_nbs, timesteps, sample_nbs_i, sample_nbs_j = jnp.meshgrid(jnp.arange(weights_seqs.shape[0]),
    #                                             jnp.arange(weights_seqs.shape[1]),
    #                                             jnp.arange(weights_seqs.shape[2]),
    #                                             jnp.arange(weights_seqs.shape[3]))

    # weights_seqs = jnp.vstack((seq_nbs.ravel(),
    #                           timesteps.ravel(),
    #                           sample_nbs_i.ravel(),
    #                           sample_nbs_j.ravel(),
    #                           weights_seqs.ravel())).T

    # weights_seqs = pd.DataFrame({'Seq nb':weights_seqs[:,0].astype(int),
    #                             'Timestep':weights_seqs[:,1].astype(int),
    #                             'Sample i': weights_seqs[:,2].astype(int),
    #                             'Sample j': weights_seqs[:,3].astype(int),
    #                             'Value':weights_seqs[:,4].tolist()})

    # print('Generating boxplots for weights...')
    # sns.violinplot(data=weights_seqs, x='Timestep', y='Value')
    # # plt.tight_layout()
    # # plt.autoscale(True)
    # plt.savefig(os.path.join(args.save_dir,'weights'))
    # plt.close()


if __name__ == '__main__':

    args = argparse.Namespace()

    args.state_dim = 2
    args.obs_dim = 2
    args.num_samples = 200
    args.num_seqs = 100
    args.seq_length = 50
    args.trained_model = False
    args.save_dir = 'experiments/tests/online/online_v1_vs_v2_200_dx=2_dy=2_N=200'
    args.exp_dir = 'experiments/p_nonlinear/2022_07_27__12_21_27'
    args.method_name = 'johnson_freeze__theta'
    args.evolution_wrt_seq_length = -1
    args.evolution_wrt_num_samples = -1
    args.evolution_wrt_dim = -1
    args.compare_online_versions = True
    args.offline_version_only = False

    if args.trained_model:

        args_p = utils.load_args('train_args', os.path.join(
            args.exp_dir, args.method_name))

        method_dir = os.path.join(args.exp_dir, args.method_name)
        args_q = utils.load_args('train_args', method_dir)
        args_p.linear = args_q.split('_')[0] == 'linear'

    else:

        args_p = argparse.Namespace()
        args_p.state_dim, args_p.obs_dim = args.state_dim, args.obs_dim
        args_p.emission_bias = False
        args_p.p_version = 'linear'
        args_p.layers = ()
        args_p.slope = 0
        args_p.num_particles = 1000
        args_p.num_smooth_particles = 1000
        args_p.transition_bias = False
        args_p.range_transition_map_params = (0.8, 0.9)
        args_p.transition_matrix_conditionning = 'diagonal'
        args_p.injective = True

        args_p.parametrization = 'cov_chol'
        args_p.default_prior_mean = 0.0
        args_p.default_prior_base_scale = math.sqrt(1e-2)
        args_p.default_transition_base_scale = math.sqrt(1e-2)
        args_p.default_emission_base_scale = math.sqrt(1e-2)
        args_p.default_transition_bias = 0

        args_q = argparse.Namespace()
        args_q.state_dim, args_q.obs_dim = args.state_dim, args.obs_dim
        args_q.q_version = 'linear'
        args_q.transition_matrix_conditionning = 'diagonal'
        args_q.range_transition_map_params = (0.99, 1)
        args_q.transition_bias = True
        args_q.emission_bias = False
        args_q.update_layers = (8,)
        args_q.backwd_layers = False
        args_q.explicit_proposal = False

    os.makedirs(args.save_dir, exist_ok=True)

    utils.save_args(args, 'args', args.save_dir)
    utils.save_args(args_p, 'args_p', args.save_dir)
    utils.save_args(args_q, 'args_q', args.save_dir)
    main(args, args_p, args_q)

# weights = jnp.exp(log_weights) / num_samples

# for t, weights_t in enumerate(weights):s
#     g = sns.displot(weights_t.flatten(), bins=100, kind='hist')
#     g.savefig(os.path.join(save_dir, f'{t}'))
