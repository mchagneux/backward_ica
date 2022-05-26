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
                            range_transition_map_params=args.range_transition_map_params,
                            transition_bias=args.transition_bias,
                            emission_bias=args.emission_bias) 
                        
    key, subkey = jax.random.split(key_theta, 2)
    theta = p.get_random_params(subkey)
    utils.save_params(theta, 'theta', save_dir)

    key, subkey = jax.random.split(key, 2)
    obs_seqs = p.sample_multiple_sequences(subkey, theta, args.num_seqs, args.seq_length)[1]

    check_linear_gaussian_elbo(p, args.num_seqs, args.seq_length)
    print('Computing evidence...')
    avg_evidence = jnp.mean(jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, theta))(obs_seqs))
    print('Avg evidence:', avg_evidence)


    q = hmm.LinearGaussianHMM(args.state_dim, 
                args.obs_dim, 
                'diagonal',
                range_transition_map_params=args.range_transition_map_params,
                transition_bias=args.transition_bias,
                emission_bias=args.emission_bias)


    trainer = SVITrainer(p=p, 
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        schedule=args.schedule)

    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi, 3)
    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                                                                        obs_seqs, 
                                                                        args.num_fits, 
                                                                        theta, 
                                                                        store_every=args.store_every) 

    utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True)
    utils.save_params(params, 'phi', save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    from datetime import datetime 
    experiment_name = 'q_linear'
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    save_dir = os.path.join(os.path.join('experiments/p_linear', 'trainings', experiment_name, date))
    os.makedirs(save_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--args_path', type=str, default='')
    args = parser.parse_args()

    if args.args_path != '':
        args = utils.load_args('train_args',args.args_path)

    else: 
        args.seed_theta = 1326 # base pseudo random key for the data model parameters
        args.seed_phi = 4569 # base pseudo random key for the variational parameters

        args.state_dim, args.obs_dim = 1,1 # dimensions of the state and observation spaces
        args.transition_matrix_conditionning = 'diagonal' # constraint on the transition matrix of the linear model

        args.seq_length = 64 # length of sequences used at training
        args.num_seqs = 4096 # number of sequences in the training set

        args.optimizer = 'adam' # optimizer for stochastic gradient descent
        args.batch_size = 64 # number of sequences in each batch
        args.parametrization = 'cov_chol' # default parametrization for covariances matrices: the gradients are performed on the cholesky matrix
        args.learning_rate = 1e-2 #{'std':1e-2, 'nn':1e-1} # learning rate for stochastic gradient descent
        args.num_epochs = 100 # number of epochs for stochastic gradient descent
        args.schedule = {} #{300:0.1} # steps to decrease the learning rate by a given factor
        args.store_every = args.num_epochs // 5 # steps at which to store the parameters
        args.num_fits = 5 # number of independent fits 
        # import math
        args.range_transition_map_params = [-1,1] # range for the parameters of the transition matrix
        args.default_prior_base_scale = 5e-2 # diagonal components of the prior covariance matrix for the data model
        args.default_transition_base_scale = 5e-2 # diagonal components of the transition covariance matrix for the data model
        args.default_emission_base_scale = 2e-2 # diagonal components of the transition covariance matrix for the data model
        args.default_prior_mean = 0.5 # mean of the initial distribution for the data model
        args.default_transition_bias = None # default bias values of the transition model
        args.transition_bias = True # bias in the transition model
        args.emission_bias = False # bias in the emission model

    args.save_dir = save_dir

    utils.save_args(args, 'train_args', save_dir)
    main(args, save_dir)

