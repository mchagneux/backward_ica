#%%
import haiku as hk 
import jax 
import jax.numpy as jnp
import backward_ica.hmm as hmm 
import backward_ica.utils as utils 
import backward_ica.smc as smc
import seaborn as sns
import os 
import matplotlib.pyplot as plt

exp_dir = 'experiments/p_nonlinear/p_nonlinear_dim_2_2/trainings/nonlinear_general_explicit_backward_with_help/2022_06_29__14_13_22'
eval_dir = os.path.join(exp_dir, 'visual_eval')

# shutil.rmtree(eval_dir)
os.makedirs(eval_dir, exist_ok=True)


args = utils.load_args('train_args',exp_dir)

utils.set_global_cov_mode(args)

key_theta = jax.random.PRNGKey(args.seed_theta)
key_phi = jax.random.PRNGKey(args.seed_phi)
num_particles = 1000
p = hmm.NonLinearGaussianHMM(state_dim=args.state_dim, 
                        obs_dim=args.obs_dim, 
                        transition_matrix_conditionning=args.transition_matrix_conditionning,
                        layers=args.emission_map_layers,
                        slope=args.slope,
                        num_particles=num_particles,
                        transition_bias=args.transition_bias,
                        range_transition_map_params=args.range_transition_map_params,
                        injective=args.injective) # specify the structure of the true model
                        
theta_star = utils.load_params('theta', exp_dir)

if args.q_version == 'linear':

    q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim,
                            transition_matrix_conditionning=args.transition_matrix_conditionning,
                            transition_bias=args.transition_bias, 
                            emission_bias=False)

elif 'nonlinear_johnson' in args.q_version:
    q = hmm.JohnsonBackwardSmoother(state_dim=args.state_dim, 
                                    obs_dim=args.obs_dim, 
                                    update_layers=args.update_layers,
                                    transition_bias=args.transition_bias)

else:
    q = hmm.GeneralBackwardSmoother(state_dim=args.state_dim, 
                                    obs_dim=args.obs_dim, 
                                    update_layers=args.update_layers,
                                    backwd_layers=args.backwd_map_layers,
                                    transition_kernel=p.transition_kernel if args.freeze_subset_params else None)


phi = utils.load_params('phi', exp_dir)

key_theta, key_gen, key_ffbsi = jax.random.split(key_theta,3)
state_seq, obs_seq = p.sample_seq(key_gen, theta_star, 200)

filt_weights, filt_particles = p.filt_seq(key_ffbsi, obs_seq, theta_star)
smoothing_paths = p.smooth_seq(key_ffbsi, obs_seq, theta_star)

filt_dir = os.path.join(eval_dir, 'filt')
smooth_dir = os.path.join(eval_dir, 'smooth')
os.makedirs(filt_dir, exist_ok=True)
os.makedirs(smooth_dir, exist_ok=True)



means_filt_ffbsi = jax.vmap(lambda particles, weights: jnp.average(a=particles, axis=0, weights=weights))(filt_particles, filt_weights)
covs_filt_ffbsi = jax.vmap(lambda mean, particles, weights: jnp.average(a=(particles-mean)**2, axis=0, weights=weights))(means_filt_ffbsi, filt_particles, filt_weights)
means_smooth_ffbsi, covs_smooth_ffbsi = jnp.mean(smoothing_paths, axis=1), jnp.var(smoothing_paths, axis=1)

key_phi, key_filt_q, key_smooth_q = jax.random.split(key_phi, 3)


if isinstance(q, hmm.GeneralBackwardSmoother):
    means_smooth_q, covs_smooth_q = q.smooth_seq(key_smooth_q, obs_seq, phi, 1000)
else:     
    means_smooth_q, covs_smooth_q = q.smooth_seq(obs_seq, phi)

if isinstance(q, hmm.GeneralBackwardSmoother) and (not q.backward_help):
    means_filt_q, covs_filt_q = q.filt_seq(key_filt_q, obs_seq, phi, 1000)
else:     
    means_filt_q, covs_filt_q = q.filt_seq(obs_seq, phi)
#%%
# bins=100
# for timestep in range(len(filt_weights)):
#     smoothing_particles_t = smoothing_paths[timestep]
#     filt_weights_t = filt_weights[timestep]
#     filt_particles_t = filt_particles[timestep]

#     if args.state_dim == 2:
#         g = sns.JointGrid(x=filt_particles_t[:,0], y=filt_particles_t[:,1], xlim=(-1.5, 1.5), ylim=(-1.5,1.5))

#         g.plot_marginals(sns.histplot, weights=filt_weights_t, bins=bins, binrange=(-1.5, 1.5))
#         g.plot_joint(sns.kdeplot, weights=filt_weights_t, fill=True)

#         g.savefig(os.path.join(filt_dir, str(timestep)))

#         g = sns.JointGrid(x=smoothing_particles_t[:,0], y=smoothing_particles_t[:,1])
#         g.plot_joint(sns.kdeplot, fill=True)
#         g.plot_marginals(sns.histplot, bins=bins, binrange=(-1.5, 1.5))
#         g.savefig(os.path.join(smooth_dir, str(timestep)))

#     elif args.state_dim == 1: 
#         g = sns.displot(x=filt_particles_t[:,0], kde=True, weights = filt_weights_t, binrange=(-1.5, 1.5), bins=bins)
#         g.savefig(os.path.join(filt_dir, str(timestep)))

#         g = sns.displot(smoothing_particles_t, kde=True, bins=bins, binrange=(-1.5, 1.5))
#         g.savefig(os.path.join(smooth_dir, str(timestep)))

#     else: 
#         pass




#%%
import numpy as np

means_ffbsi = [means_smooth_ffbsi, means_filt_ffbsi]
covs_ffbsi = [covs_smooth_ffbsi, covs_filt_ffbsi]
means_q = [means_smooth_q, means_filt_q]
covs_q = [covs_smooth_q, covs_filt_q]
names = ['smoothing_eval', 'filt_eval']
for mean_ffbsi, cov_ffbsi, mean_q, cov_q, name in zip(means_ffbsi, covs_ffbsi, means_q, covs_q, names):
    fig, axes = plt.subplots(args.state_dim, 2, figsize=(15,15))
    axes = np.atleast_2d(axes)

    for dim_nb in range(args.state_dim):

        utils.plot_relative_errors_1D(axes[dim_nb,0], state_seq[:,dim_nb], mean_ffbsi[:,dim_nb], cov_ffbsi[:,dim_nb])

    for dim_nb in range(args.state_dim):

        if isinstance(q, hmm.LinearBackwardSmoother) or q.backward_help:
            utils.plot_relative_errors_1D(axes[dim_nb,1], state_seq[:,dim_nb], mean_q[:,dim_nb], cov_q[:,dim_nb,dim_nb])
        else:
            utils.plot_relative_errors_1D(axes[dim_nb,1], state_seq[:,dim_nb], mean_q[:,dim_nb], cov_q[:,dim_nb])

    plt.autoscale(True)
    plt.tight_layout()
    plt.savefig(os.path.join(eval_dir, name))
    plt.clf()