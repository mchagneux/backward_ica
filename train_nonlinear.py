from argparse import ArgumentParser
from re import I
import jax 
import jax.numpy as jnp
from jax import config, tree_map 

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer
utils.enable_x64(True)
import tensorflow as tf

def main(args, save_dir):

    utils.set_global_cov_mode(args)

    key_theta = jax.random.PRNGKey(args.seed_theta)
    key_phi = jax.random.PRNGKey(args.seed_phi)

    p = hmm.NonLinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim, 
                            transition_matrix_conditionning=args.transition_matrix_conditionning,
                            layers=args.emission_map_layers,
                            slope=args.slope,
                            num_particles=args.num_particles,
                            transition_bias=args.transition_bias,
                            range_transition_map_params=args.range_transition_map_params,
                            injective=args.injective) # specify the structure of the true model
                            
    key_params, key_gen, key_smc = jax.random.split(key_theta, 3)

    theta_star = p.get_random_params(key_params) # sample params randomly (but covariances are fixed to default values)

    utils.save_params(theta_star, 'theta', save_dir)

    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, args.num_seqs, args.seq_length)


    key_smoothing, key_evidence = jax.random.split(key_smc, 2)

    # state_seq, obs_seq = state_seqs[0], obs_seqs[0]

    # means_smc, covs_smc = p.smooth_seq(key_smoothing, obs_seq, theta)

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(args.state_dim, figsize=(30,30))
    # for dim_nb in range(args.state_dim):
        
    #     utils.plot_relative_errors_1D(axes[dim_nb], state_seq[:,dim_nb], means_smc[:,dim_nb], covs_smc[:,dim_nb])
    #     utils.plot_relative_errors_1D(axes[dim_nb], state_seq[:,dim_nb], means_smc[:,dim_nb], covs_smc[:,dim_nb])

    # plt.autoscale(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, 'smc_smoothing_on_first_seq.pdf'),format='pdf')
    # plt.clf()

    evidence_keys = jax.random.split(key_evidence, args.num_seqs)

    print('Computing evidence...')

    avg_evidence = jnp.mean(jax.vmap(jax.jit(lambda key, obs_seq: p.likelihood_seq(key, obs_seq, 
                                                                        theta_star)))(evidence_keys, obs_seqs)) / args.seq_length


    print('Oracle evidence:', avg_evidence)


    if args.q_version == 'linear':

        q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                                obs_dim=args.obs_dim,
                                transition_matrix_conditionning=args.transition_matrix_conditionning,
                                transition_bias=args.transition_bias, 
                                emission_bias=False)

    elif 'nonlinear_johnson' in args.q_version:
        q = hmm.JohnsonBackwardSmoother(transition_kernel=p.transition_kernel,
                                        obs_dim=args.obs_dim, 
                                        update_layers=args.update_layers,
                                        explicit_proposal=args.explicit_proposal)

    else:
        q = hmm.GeneralBackwardSmoother(state_dim=args.state_dim, 
                                        obs_dim=args.obs_dim, 
                                        update_layers=args.update_layers,
                                        backwd_layers=args.backwd_map_layers)




    frozen_theta = p.get_random_params(key_theta)
    frozen_theta = tree_map(lambda x: '', frozen_theta)

    frozen_phi = q.get_random_params(key_phi)
    frozen_phi = tree_map(lambda x: '', frozen_phi)

    if 'theta' in args.frozen_params: 
        frozen_theta = theta_star 

    if 'prior_phi' in args.frozen_params:
        if isinstance(q, hmm.LinearGaussianHMM) or (isinstance(q, hmm.JohnsonBackwardSmoother) and q.explicit_proposal):
            frozen_phi.prior = theta_star.prior
        else:
            frozen_phi.prior = q.get_init_state()
    
    if 'transition_phi' in args.frozen_params:
        if isinstance(q, hmm.GeneralBackwardSmoother):
            raise NotImplementedError
        else: 
            frozen_phi.transition = theta_star.transition
    
    frozen_params = (frozen_theta, frozen_phi)


    trainer = SVITrainer(p=p, 
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        schedule=args.schedule,
                        num_samples=args.num_samples,
                        force_full_mc=args.full_mc,
                        frozen_params=frozen_params)


    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi, 3)

    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                                                            data=obs_seqs, 
                                                            num_fits=args.num_fits,
                                                            log_dir=save_dir) # returns the best fit (based on the last value of the elbo)
    
    utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True, best_epochs_only=True)
    utils.save_params(params, 'phi', save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    from datetime import datetime
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


    
    parser = argparse.ArgumentParser()

    parser.add_argument('--q_version',type=str, default='nonlinear_johnson')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--injective', dest='injective', action='store_true', default=True)
    parser.add_argument('--args_path', type=str, default='')
    args = parser.parse_args()



    save_dir = args.save_dir

    if save_dir=='':
        save_dir = os.path.join('experiments','tests',date)
        os.makedirs(save_dir, exist_ok=True)

    if args.args_path != '':
        args = utils.load_args('train_args',args.args_path)
        args.save_dir = save_dir
    else:
        args.seed_theta = 1329
        args.seed_phi = 4569

        args.state_dim, args.obs_dim = 2,2
        args.transition_matrix_conditionning = 'diagonal'

        args.emission_map_layers = ()
        args.slope = 0


        args.seq_length = 50
        args.num_seqs = 4096


        args.optimizer = 'adam'
        args.batch_size = 512
        args.parametrization = 'cov_chol'
        args.learning_rate = 1e-2 # {'std':1e-2, 'nn':1e-1}
        args.num_epochs = 300
        args.schedule = {} # {20:0.1} #{'nn':{200:0.1, 250:0.5}}
        args.store_every = args.num_epochs // 5
        args.num_fits = 1

        args.update_layers = (8,8)
        args.backwd_map_layers = (8,8)


        args.num_particles = 1000
        args.num_samples = 1
        args.parametrization = 'cov_chol'
        import math
        args.default_prior_mean = 0.0
        args.range_transition_map_params = [0.99,1]
        args.default_prior_base_scale = math.sqrt(1e-2)
        args.default_transition_base_scale = math.sqrt(1e-2)
        args.default_emission_base_scale = math.sqrt(1e-3)
        args.default_transition_bias = 0
        args.transition_bias = False
        args.full_mc = 'full_mc' in args.q_version
        args.explicit_proposal = 'explicit_proposal' in args.q_version 
        args.frozen_params  = args.q_version.split('__')[1:]        

    utils.save_args(args, 'train_args', save_dir)

    main(args, save_dir)
