import jax 
import jax.numpy as jnp

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer, check_linear_gaussian_elbo
utils.enable_x64(True)

def main(args, save_dir):
    key_theta = jax.random.PRNGKey(args.seed_theta)
    key_phi = jax.random.PRNGKey(args.seed_phi)

    hmm.HMM.parametrization = args.parametrization 

    p = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim, 
                            transition_matrix_conditionning=args.transition_matrix_conditionning) # specify the structure of the true model
                        
    key, subkey = jax.random.split(key_theta, 2)
    theta = p.get_random_params(subkey) # sample params randomly (but covariances are fixed to default values)
    utils.save_params(theta, 'theta', save_dir)

    key, subkey = jax.random.split(key, 2)
    obs_seqs = p.sample_multiple_sequences(subkey, theta, args.num_seqs, args.seq_length)[1]

    # check_linear_gaussian_elbo(obs_seqs, p, theta)
    print('Computing evidence...')
    avg_evidence = jnp.mean(jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, theta))(obs_seqs))
    print('Avg evidence:', avg_evidence)

    # mle_fit, mle_curve = p.fit_kalman_rmle(key, obs_seqs, args.optimizer, 1e-2, args.batch_size, 3)
    # import matplotlib.pyplot as plt
    # plt.plot(mle_curve)
    # plt.show()

    # q = hmm.LinearGaussianHMM(args.state_dim, args.obs_dim, 'diagonal')
    
    q = hmm.NeuralLinearBackwardSmoother(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim, 
                            transition_kernel_matrix_conditionning=args.transition_matrix_conditionning,
                            use_johnson=True,
                            update_layers=(8,)) # specify the structure of the true model, but init params are sampled during optimisiation     

    trainer = SVITrainer(p=p, 
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        schedule=args.schedule,
                        force_full_mc=True)
    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi, 3)
    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                                                                        obs_seqs, 
                                                                        args.num_fits, 
                                                                        theta, 
                                                                        store_every=None) # returns the best fit (based on the last value of the elbo)
    utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True)
    utils.save_params(params, f'phi_every_{args.store_every}_epochs', save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    args = argparse.Namespace()

    experiment_name = 'test_johnson'
    save_dir = os.path.join(os.path.join('experiments', experiment_name))
    os.mkdir(save_dir)


    args.seed_theta = 1326
    args.seed_phi = 4569

    args.state_dim, args.obs_dim = 1,1
    args.transition_matrix_conditionning = 'diagonal'

    args.seq_length = 8
    args.num_seqs = 12800

    args.optimizer = 'adam'
    args.batch_size = 64
    args.parametrization = 'cov_chol'
    args.learning_rate = 1e-2 #{'std':1e-2, 'nn':1e-1}
    args.num_epochs = 600
    args.schedule = {} #{300:0.1}
    args.store_every = args.num_epochs // 5
    args.num_fits = 2

    utils.save_args(args, 'train_args', save_dir)
    main(args, save_dir)

