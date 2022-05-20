from argparse import ArgumentParser
import jax 
import jax.numpy as jnp
from jax import config 

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer
utils.enable_x64(True)


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
                            injective=args.injective) # specify the structure of the true model
    key_params, key_gen, key_smc = jax.random.split(key_theta, 3)

    theta = p.get_random_params(key_params) # sample params randomly (but covariances are fixed to default values)
    utils.save_params(theta, 'theta', save_dir)

    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta, args.num_seqs, args.seq_length)

    import matplotlib.pyplot as plt 


        # for state_seq in state_seqs: 
        #     plt.scatter(range(len(state_seq)), state_seq)
        # plt.savefig(os.path.join(save_dir,'example_states'))
        # plt.clf()
    # state_seqs_1st_coord = state_seqs[:100]
    # for state_seq in state_seqs_1st_coord: 
    #     plt.plot(state_seq, linestyle='dotted', marker='.')
    # plt.savefig(os.path.join(save_dir,'example_states'))
    # plt.clf()

    # mapped_state_seqs_1st_coord = p.emission_kernel.map(state_seqs_1st_coord, theta.emission).mean
    # for mapped_state_seq in mapped_state_seqs_1st_coord:
    #     plt.plot(mapped_state_seq, linestyle='dotted', marker='.')
    # plt.savefig(os.path.join(save_dir,'example_mapped_states'))
    # plt.clf()

    support = jnp.sort(state_seqs.flatten()).reshape(-1,1)
    plt.plot(support, p.emission_kernel.map(support, theta.emission).mean)
    plt.savefig(os.path.join(save_dir,'emission_map_on_states_support'))
    plt.clf()
    

    # support = jnp.sort(state_seqs[0].flatten()).reshape(-1,1)
    # plt.plot(support, p.emission_kernel.map(support, theta.emission).mean)
    # plt.savefig(os.path.join(save_dir,'emission_map_on_single_sequence'))
    # plt.clf()
    support_stationary = jnp.sort(state_seqs[:,2:,:].flatten()).reshape(-1,1)
    plt.plot(support_stationary, p.emission_kernel.map(support_stationary, theta.emission).mean)
    plt.savefig(os.path.join(save_dir,'emission_map_on_states_support_stationary'))
    plt.clf()

    smc_keys = jax.random.split(key_smc, args.num_seqs)

    print('Computing evidence...')

    avg_evidence = jnp.mean(jax.vmap(jax.jit(lambda obs_seq, key: p.likelihood_seq(key, obs_seq, 
                                                                        theta)))(obs_seqs, smc_keys))


    print('Avg evidence:', avg_evidence)


    if args.q_version == 'linear':

        q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                                obs_dim=args.obs_dim,
                                transition_matrix_conditionning=args.transition_matrix_conditionning,
                                transition_bias=args.transition_bias, emission_bias=False)

    else: 
        version = args.q_version.split('_')[1]
        q = hmm.NeuralLinearBackwardSmoother(state_dim=args.state_dim, 
                                        obs_dim=args.obs_dim, 
                                        use_johnson=(version == 'johnson'),
                                        update_layers=args.update_layers,
                                        transition_bias=args.transition_bias)

    # q = hmm.NeuralBackwardSmoother(state_dim=args.state_dim, 
    #                         obs_dim=args.obs_dim, 
    #                         update_layers=args.update_layers,
    #                         backwd_layers=args.backwd_map_layers)


    trainer = SVITrainer(p=p, 
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        schedule=args.schedule,
                        num_samples=args.num_samples,
                        force_full_mc=True)

    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi,3)

    # phi, avg_logls = q.fit_kalman_rmle(key_params, obs_seqs, args.optimizer, args.learning_rate, args.batch_size, args.num_epochs)
    # print(avg_logls[-1])
    # plt.plot(avg_logls)
    # plt.show()

    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                                                            data=obs_seqs, 
                                                            theta_star=theta, 
                                                            num_fits=args.num_fits) # returns the best fit (based on the last value of the elbo)
    
    utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True, best_epochs_only=True)
    utils.save_params(params, 'phi', save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    from datetime import datetime
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--q_version',type=str, default='nonlinear_johnson')
    parser.add_argument('--save_dir', type=str, default='test')
    parser.add_argument('--injective', dest='injective', action='store_true', default=False)

    args = parser.parse_args()
    # args.injective = True

    save_dir = args.save_dir
    
    # sys.stdout = open(os.path.join(save_dir, 'train_logs.txt'), 'w')

    args.seed_theta = 1329
    args.seed_phi = 4569

    args.state_dim, args.obs_dim = 5,5
    args.transition_matrix_conditionning = 'diagonal'
    args.emission_map_layers = () 
    args.slope = 0


    args.seq_length = 4
    args.num_seqs = 12800


    args.optimizer = 'adam'
    args.batch_size = 64
    args.parametrization = 'cov_chol'
    args.learning_rate = 1e-3 # {'std':1e-2, 'nn':1e-1}
    args.num_epochs = 300
    args.schedule = {} #{'nn':{200:0.1, 250:0.5}}
    args.store_every = args.num_epochs // 5
    args.num_fits = 6
    
    args.update_layers = (16,16)
    args.backwd_map_layers = (16,16)


    args.num_particles = 1000
    args.num_samples = 10
    args.parametrization = 'cov_chol'
    import math
    args.default_prior_base_scale = math.sqrt(1e-2)
    args.default_transition_base_scale = math.sqrt(1e-2)
    args.default_emission_base_scale = math.sqrt(1e-2)
    args.default_transition_bias = 0.3
    args.transition_bias = True

    utils.save_args(args, 'train_args', save_dir)

    main(args, save_dir)
