import jax 
import jax.numpy as jnp
from jax import config
import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.elbos import SVITrainer, check_linear_gaussian_elbo
config.update('jax_enable_x64',True)

def main(args, save_dir):

    key_theta = jax.random.PRNGKey(args.seed_theta)
    key_phi = jax.random.PRNGKey(args.seed_phi)

    utils.set_defaults(args)

    p = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim, 
                            transition_matrix_conditionning=args.transition_matrix_conditionning,
                            range_transition_map_params=args.range_transition_map_params,
                            transition_bias=args.transition_bias,
                            emission_bias=args.emission_bias) 
                        
    key, subkey = jax.random.split(key_theta, 2)
    theta_star = p.get_random_params(subkey)
    utils.save_params(theta_star, 'theta', save_dir)

    key, subkey = jax.random.split(key, 2)
    obs_seqs = p.sample_multiple_sequences(subkey, theta_star, args.num_seqs, args.seq_length)[1]

    check_linear_gaussian_elbo(p, args.num_seqs, args.seq_length)
    print('Computing evidence...')
    avg_evidence = jnp.mean(jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, theta_star))(obs_seqs)) / args.seq_length
    print('Avg evidence:', avg_evidence)


    q = hmm.LinearGaussianHMM(args.state_dim, 
                args.obs_dim, 
                'diagonal',
                range_transition_map_params=(0,1),
                transition_bias=args.transition_bias,
                emission_bias=args.emission_bias)

    frozen_params = utils.define_frozen_tree(key_phi, args.frozen_params, p, q, theta_star)


    trainer = SVITrainer(p=p, 
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        num_samples=0,
                        force_full_mc=args.full_mc,
                        frozen_params=frozen_params,
                        online=False)

    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi, 3)

    params, avg_elbos = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                            data=obs_seqs, 
                            num_fits=args.num_fits,
                            log_dir=save_dir,
                            store_every=args.store_every)

    utils.save_params(params, 'phi', save_dir)
    import dill
    with open(os.path.join(save_dir, 'train_logs'), 'wb') as f: 
        dill.dump((avg_evidence, avg_elbos), f)
    # phi = params[-1]

    # state_seq, obs_seq = p.sample_seq(key, theta, 100)



    # means_star, covs_star = p.smooth_seq(obs_seq, theta)
    # means_mle, covs_mle = q.smooth_seq(obs_seq, phi)

    # import matplotlib.pyplot as plt 
    # fig, axes = plt.subplots(args.state_dim,2, figsize=(15,15))
    # import numpy as np
    # axes = np.atleast_2d(axes)

    # for dim_nb in range(args.state_dim):
        
    #     utils.plot_relative_errors_1D(axes[dim_nb,0], state_seq[:,dim_nb], means_star[:,dim_nb], covs_star[:,dim_nb,dim_nb])
    #     utils.plot_relative_errors_1D(axes[dim_nb,1], state_seq[:,dim_nb], means_mle[:,dim_nb], covs_mle[:,dim_nb,dim_nb])

    # plt.autoscale(True)
    # plt.tight_layout()
    # plt.savefig(os.path.join(save_dir, 'example_smoothed_states'))

if __name__ == '__main__':

    import argparse
    import os 
    from datetime import datetime 
    experiment_name = 'q_linear'
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

    save_dir = os.path.join(os.path.join('experiments/p_linear', date))
    os.makedirs(save_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--args_path', type=str, default='')
    args = parser.parse_args()

    if args.args_path != '':
        args = utils.load_args('train_args',args.args_path)

    else: 
        args.seed_theta = 0
        args.seed_phi = 1
        args.frozen_params = ['theta']

        args.state_dim, args.obs_dim = 5,5
        args.transition_matrix_conditionning = 'diagonal'

        args.seq_length = 50
        args.num_seqs = 1000

        args.optimizer = 'adam'
        args.batch_size = args.num_seqs // 100
        args.learning_rate = 1e-3 #{'std':1e-2, 'nn':1e-1}
        args.num_epochs = 200
        args.schedule = {} #{300:0.1}
        args.store_every = 5
        args.num_fits = 1

        args.parametrization = 'cov_chol'
        import math
        args.default_prior_mean = 0.0
        args.range_transition_map_params = [0.99,1]
        args.default_prior_base_scale = math.sqrt(1e-2)
        args.default_transition_base_scale = math.sqrt(1e-2)
        args.default_emission_base_scale = math.sqrt(1e-3)
        args.default_transition_bias = 0
        args.transition_bias = False
        args.emission_bias = False
        args.full_mc = False

    args.save_dir = save_dir

    utils.save_args(args, 'train_args', save_dir)
    main(args, save_dir)

