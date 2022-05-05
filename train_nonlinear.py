import jax 
import jax.numpy as jnp
from jax import config 

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer
utils.enable_x64(True)


def main(args, save_dir):


    key_theta = jax.random.PRNGKey(args.seed_theta)
    key_phi = jax.random.PRNGKey(args.seed_phi)

    p = hmm.NonLinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim, 
                            transition_matrix_conditionning=args.transition_matrix_conditionning,
                            layers=args.emission_map_layers,
                            slope=args.slope,
                            num_particles=args.num_particles) # specify the structure of the true model

    key_params, key_gen, key_smc = jax.random.split(key_theta, 3)

    theta = p.get_random_params(key_params) # sample params randomly (but covariances are fixed to default values)
    utils.save_params(theta, 'theta', save_dir)

    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta, args.num_seqs, args.seq_length)

    import matplotlib.pyplot as plt 


    if args.state_dim == 1: 
        # for state_seq in state_seqs: 
        #     plt.scatter(range(len(state_seq)), state_seq)
        # plt.savefig(os.path.join(save_dir,'example_states'))
        # plt.clf()
        for state_seq in state_seqs[:100]: 
            plt.plot(state_seq, linestyle='dotted', marker='.')
        plt.savefig(os.path.join(save_dir,'example_states'))
        plt.clf()

    if args.obs_dim == 1: 
        mapped_state_seqs = p.emission_kernel.map(state_seqs[:100], theta.emission).mean
        for mapped_state_seq in mapped_state_seqs:
            plt.plot(mapped_state_seq, linestyle='dotted', marker='.')
        plt.savefig(os.path.join(save_dir,'example_mapped_states'))
        plt.clf()

        support = jnp.sort(state_seqs.flatten()).reshape(-1,1)
        plt.plot(support, p.emission_kernel.map(support, theta.emission).mean)
        plt.savefig(os.path.join(save_dir,'emission_map_on_states_support'))
        plt.clf()
        
        support_stationary = jnp.sort(state_seqs[:,3:,:].flatten()).reshape(-1,1)
        plt.plot(support_stationary, p.emission_kernel.map(support_stationary, theta.emission).mean)
        plt.savefig(os.path.join(save_dir,'emission_map_on_states_support_stationary'))
        plt.clf()


    smc_keys = jax.random.split(key_smc, args.num_seqs)

    print('Computing evidence...')

    avg_evidence = jnp.mean(jax.vmap(jax.jit(lambda obs_seq, key: p.likelihood_seq(key, obs_seq, 
                                                                        theta)))(obs_seqs, smc_keys))


    print('Avg evidence:', avg_evidence)


    # q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
    #                         obs_dim=args.obs_dim,
    #                         transition_matrix_conditionning=args.transition_matrix_conditionning)

    q = hmm.NeuralLinearBackwardSmoother(state_dim=args.state_dim, 
                                    obs_dim=args.obs_dim, 
                                    use_johnson=True,
                                    update_layers=args.update_layers)

    # q = hmm.NeuralBackwardSmoother(state_dim=args.state_dim, 
    #                         obs_dim=args.obs_dim, 
    #                         update_layers=args.update_layers,
    #                         backwd_layers=args.backwd_map_layers)

    trainer = SVITrainer(p, q, 
                        args.optimizer, 
                        args.learning_rate, 
                        args.num_epochs, 
                        args.batch_size, 
                        args.num_samples, 
                        force_mc=False)

    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(*jax.random.split(key_phi,3), obs_seqs, theta, args.num_fits) # returns the best fit (based on the last value of the elbo)
    utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True)
    utils.save_params(params, 'phi', save_dir)

if __name__ == '__main__':

    import argparse
    import os 

    args = argparse.Namespace()

    experiment_name = 'nonlinear_p_nonlinear_q'
    save_dir = os.path.join(os.path.join('experiments', experiment_name))
    
    os.mkdir(save_dir)

    # sys.stdout = open(os.path.join(save_dir, 'train_logs.txt'), 'w')

    args.seed_theta = 1329
    args.seed_phi = 4569

    args.state_dim, args.obs_dim = 1,1 
    args.transition_matrix_conditionning = 'diagonal'
    args.emission_map_layers = ()
    args.slope = 0

    args.seq_length = 8
    args.num_seqs = 12800

    args.optimizer = 'adam'
    args.batch_size = 64
    args.learning_rate = 1e-2
    args.num_epochs = 300
    args.store_every = args.num_epochs // 5
    args.num_fits = 5
    
    args.update_layers = (16,16)
    args.backwd_map_layers = (16,16)


    args.num_particles = 1000
    args.num_samples = 1


    utils.save_args(args, 'train_args', save_dir)
    main(args, save_dir)
