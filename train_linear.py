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
                range_transition_map_params=(0,1),
                transition_bias=args.transition_bias,
                emission_bias=args.emission_bias)


    trainer = SVITrainer(p=p, 
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        schedule=args.schedule,
                        froze_subset_params=True)

    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi, 3)
    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                                                                        obs_seqs, 
                                                                        args.num_fits, 
                                                                        theta, 
                                                                        store_every=args.store_every,
                                                                        log_dir=save_dir) 

    utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True)
    utils.save_params(params, 'phi', save_dir)

    phi = params[-1]

    state_seq, obs_seq = p.sample_seq(key, theta, 100)



    means_star, covs_star = p.smooth_seq(obs_seq, theta)
    means_mle, covs_mle = q.smooth_seq(obs_seq, phi)

    import matplotlib.pyplot as plt 
    fig, axes = plt.subplots(args.state_dim,2, figsize=(15,15))
    import numpy as np
    axes = np.atleast_2d(axes)

    for dim_nb in range(args.state_dim):
        
        utils.plot_relative_errors_1D(axes[dim_nb,0], state_seq[:,dim_nb], means_star[:,dim_nb], covs_star[:,dim_nb,dim_nb])
        utils.plot_relative_errors_1D(axes[dim_nb,1], state_seq[:,dim_nb], means_mle[:,dim_nb], covs_mle[:,dim_nb,dim_nb])

    plt.autoscale(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'example_smoothed_states'))

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
        args.seed_theta = 0
        args.seed_phi = 1

        args.state_dim, args.obs_dim = 1,1
        args.transition_matrix_conditionning = 'diagonal'

        args.seq_length = 10
        args.num_seqs = 10000

        args.optimizer = 'adam'
        args.batch_size = args.num_seqs // 100
        args.parametrization = 'cov_chol'
        args.learning_rate = 1e-3 #{'std':1e-2, 'nn':1e-1}
        args.num_epochs = 100
        args.schedule = {} #{300:0.1}
        args.store_every = args.num_epochs // 5
        args.num_fits = 5
        import math
        args.range_transition_map_params = [0.99,1]
        args.default_prior_base_scale = math.sqrt(0.1)
        args.default_transition_base_scale = math.sqrt(0.1)
        args.default_emission_base_scale = math.sqrt(0.1)
        args.default_prior_mean = 0.0
        args.default_transition_bias = None
        args.transition_bias = False
        args.emission_bias = False

    args.save_dir = save_dir

    utils.save_args(args, 'train_args', save_dir)
    main(args, save_dir)

