import jax 
import jax.numpy as jnp
from jax import config
import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer, check_linear_gaussian_elbo
config.update('jax_enable_x64',True)

def main(args, save_dir):

    key_theta = jax.random.PRNGKey(args.seed_theta)
    key_phi = jax.random.PRNGKey(args.seed_phi)

    utils.set_global_cov_mode(args)

    p = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim, 
                            transition_matrix_conditionning=args.transition_matrix_conditionning,
                            transition_bias=args.transition_bias,
                            emission_bias=args.emission_bias) # specify the structure of the true model
                        
    key, subkey = jax.random.split(key_theta, 2)
    theta = p.get_random_params(subkey) # sample params randomly (but covariances are fixed to default values)
    utils.save_params(theta, 'theta', save_dir)

    key, subkey = jax.random.split(key, 2)
    obs_seqs = p.sample_multiple_sequences(subkey, theta, args.num_seqs, args.seq_length)[1]

    check_linear_gaussian_elbo(p, args.num_seqs, args.seq_length)
    print('Computing evidence...')
    avg_evidence = jnp.mean(jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, theta))(obs_seqs))
    print('Avg evidence:', avg_evidence)

    # mle_fit, mle_curve = p.fit_kalman_rmle(key, obs_seqs, args.optimizer, 1e-2, args.batch_size, 3)
    # import matplotlib.pyplot as plt
    # plt.plot(mle_curve)
    # plt.show()

    q = hmm.LinearGaussianHMM(args.state_dim, 
                args.obs_dim, 
                'diagonal',
                transition_bias=args.transition_bias,
                emission_bias=args.emission_bias)


    trainer = SVITrainer(p=p, 
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        schedule=args.schedule,
                        force_full_mc=False)

    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi, 3)
    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                                                                        obs_seqs, 
                                                                        args.num_fits, 
                                                                        theta, 
                                                                        store_every=args.store_every) 
                                                                        # returns the best fit (based on the last value of the elbo)
    utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True)
    utils.save_params(params, 'phi', save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    from datetime import datetime 
    
    args = argparse.Namespace()

    experiment_name = 'q_linear'
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    save_dir = os.path.join(os.path.join('experiments/p_linear', 'trainings', experiment_name, date))
    os.makedirs(save_dir, exist_ok=True)


    args.seed_theta = 1326
    args.seed_phi = 4569

    args.state_dim, args.obs_dim = 5,5
    args.transition_matrix_conditionning = 'diagonal'

    args.seq_length = 8
    args.num_seqs = 6400

    args.optimizer = 'adam'
    args.batch_size = 64
    args.parametrization = 'cov_chol'
    args.learning_rate = 1e-3 #{'std':1e-2, 'nn':1e-1}
    args.num_epochs = 300
    args.schedule = {} #{300:0.1}
    args.store_every = args.num_epochs // 5
    args.num_fits = 5
    args.save_dir = save_dir
    import math
    
    args.default_prior_base_scale = math.sqrt(1e-1)
    args.default_transition_base_scale = math.sqrt(1e-2)
    args.default_emission_base_scale = math.sqrt(1e-2)
    args.default_transition_bias = 0.5
    args.transition_bias = False
    args.emission_bias = False

    utils.save_args(args, 'train_args', save_dir)
    main(args, save_dir)

