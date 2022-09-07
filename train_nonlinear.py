import jax 
import jax.numpy as jnp
import math

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer
utils.enable_x64(True)

def main(args, save_dir):

    utils.set_parametrization(args)

    key_theta = jax.random.PRNGKey(args.seed_theta)
    key_phi = jax.random.PRNGKey(args.seed_phi)

    if args.p_version == 'linear':
        p = hmm.LinearGaussianHMM(args.state_dim, 
                                args.obs_dim, 
                                args.transition_matrix_conditionning, 
                                args.range_transition_map_params,
                                args.transition_bias, 
                                args.emission_bias)
    elif 'choatic_rnn' in args.p_version:
        p = hmm.NonLinearHMM.chaotic_rnn(args)
    else: 
        p = hmm.NonLinearHMM.linear_transition_with_nonlinear_emission(args) # specify the structure of the true model
                            
    key_params, key_gen, key_smc = jax.random.split(key_theta, 3)

    theta_star = p.get_random_params(key_params, args) # sample params randomly (but covariances are fixed to default values)

    utils.save_params(theta_star, 'theta', save_dir)

    _ , obs_seqs = p.sample_multiple_sequences(key_gen, 
                                                    theta_star, 
                                                    args.num_seqs, 
                                                    args.seq_length, 
                                                    single_split_seq=args.single_split_seq)



    evidence_keys = jax.random.split(key_smc, args.num_seqs)

    print('Computing evidence...')

    avg_evidence = jnp.mean(jax.vmap(jax.jit(lambda key, obs_seq: p.likelihood_seq(key, obs_seq, 
                                                                        theta_star)))(evidence_keys, obs_seqs)) / args.seq_length


    print('Oracle evidence:', avg_evidence)


    if 'linear' in args.q_version:

        q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                                obs_dim=args.obs_dim,
                                transition_matrix_conditionning=args.transition_matrix_conditionning,
                                transition_bias=args.transition_bias, 
                                emission_bias=False)

    elif 'johnson' in args.q_version:
        q = hmm.JohnsonBackwardSmoother(transition_kernel=p.transition_kernel,
                                        obs_dim=args.obs_dim, 
                                        update_layers=args.update_layers,
                                        explicit_proposal=args.explicit_proposal)

    else:
        q = hmm.GeneralBackwardSmoother(state_dim=args.state_dim, 
                                        obs_dim=args.obs_dim, 
                                        update_layers=args.update_layers,
                                        backwd_layers=args.backwd_map_layers)




    frozen_params = utils.define_frozen_tree(key_phi, args.frozen_params, p, q, theta_star)


    trainer = SVITrainer(p=p, 
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        num_samples=args.num_samples,
                        force_full_mc=args.full_mc,
                        frozen_params=frozen_params,
                        online=args.online)


    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi, 3)

    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                                                            data=obs_seqs, 
                                                            num_fits=args.num_fits,
                                                            log_dir=save_dir,
                                                            args=args) # returns the best fit (based on the last value of the elbo)
    
    utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True, best_epochs_only=True)
    utils.save_params(params, 'phi', save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    from datetime import datetime
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


    
    parser = argparse.ArgumentParser()

    parser.add_argument('--p_version', type=str, default='chaotic_rnn')
    parser.add_argument('--q_version',type=str, default='johnson_freeze__theta')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--injective', dest='injective', action='store_true', default=True)
    parser.add_argument('--args_path', type=str, default='')
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--dims', type=int, nargs='+', default=(2,2))


    args = parser.parse_args()



    save_dir = args.save_dir

    if save_dir=='':
        save_dir = os.path.join('experiments','tests',date)
        os.makedirs(save_dir, exist_ok=True)

    if args.args_path != '':
        args = utils.load_args('train_args',args.args_path)
        args.save_dir = save_dir
    else:

        ## randomness 
        args.seed_theta = 1329
        args.seed_phi = 4569

        ## dataset 
        args.state_dim, args.obs_dim = args.dims 
        args.seq_length = 50 # length of the train sequences
        args.num_seqs = 100 # number of train sequences
        args.single_split_seq = False # whether to draw one long sample of length seq_length * num_seqs and divide it in seq_length // num_seqs sequences


        args.parametrization = 'cov_chol' # parametrization of the covariance matrices 

        ## prior 
        args.default_prior_mean = 0.0 # default value for the mean of Gaussian prior
        args.default_prior_base_scale = math.sqrt(1e-2) # default value for the diagonal components of the covariance matrix of the prior

        ## transition 
        args.transition_matrix_conditionning = 'diagonal' # constraint on the transition matrix 
        args.range_transition_map_params = [0.99,1] # range of the components of the transition matrix
        args.default_transition_base_scale = math.sqrt(1e-2) # default value for the diagonal components of the covariance matrix of the transition kernel
        args.transition_bias = False 
        args.default_transition_bias = 0

        ## emission 
        args.emission_matrix_conditionning = 'diagonal'
        args.emission_bias = False
        args.emission_map_layers = (8,)
        args.range_emission_map_params = (0.99,1)
        args.default_emission_base_scale = math.sqrt(1e-2)
        args.default_emission_matrix = 1
        args.slope = 0 # amount of linearity in the emission function
        args.grid_size = 0.001 # discretization parameter for the chaotic rnn
        args.gamma = 2.5 # gamma for the chaotic rnn
        args.tau = 0.025 # tau for the chaotic rnn

        ## variational family
        args.explicit_proposal = 'explicit_proposal' in args.q_version # whether to use a Kalman predict step as a first move to update the variational filtering familiy
        args.update_layers = (8,8) # number of layers in the GRU which updates the variational filtering dist
        args.backwd_map_layers = (8,8) # number of layers in the MLP which predicts backward parameters (not used in the Johnson method)

        ## SMC 
        args.num_particles = 2 # number of particles for bootstrap filtering step
        args.num_smooth_particles = 2 # number of particles for the FFBSi ancestral sampling step

        ## optimizer
        args.optimizer = 'adamw' 
        args.batch_size = 100
        args.num_epochs = 1000
        args.store_every = args.num_epochs // 5 # step to store intermediate parameter values
        args.num_fits = 1 # number of optimization runs starting from multiple seeds
        args.num_samples = 1  # number of MCMC samples used to compute the ELBO
        args.full_mc = 'full_mc' in args.q_version # whether to force the use the full MCMC ELBO (e.g. prevent using closed-form terms even with linear models)
        args.frozen_params  = args.q_version.split('__')[1:] # list of parameter groups which are not learnt
        args.online = 'online' in args.q_version # whether to use the online ELBO or not

    utils.save_args(args, 'train_args', save_dir)

    main(args, save_dir)
