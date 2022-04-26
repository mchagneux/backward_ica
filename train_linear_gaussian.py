import jax 
import jax.numpy as jnp
from jax import config 

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer, check_linear_gaussian_elbo


def main(args, save_dir):
    key_theta = jax.random.PRNGKey(args.seed_theta)
    key_phi = jax.random.PRNGKey(args.seed_phi)

    p = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim, 
                            transition_matrix_conditionning=args.transition_matrix_conditionning) # specify the structure of the true model
                        
    key, subkey = jax.random.split(key_theta, 2)
    theta = p.get_random_params(subkey) # sample params randomly (but covariances are fixed to default values)
    utils.save_params(theta, 'theta', save_dir)

    key, subkey = jax.random.split(key, 2)
    obs_seqs = hmm.sample_multiple_sequences(subkey, p.sample_seq, theta, args.num_seqs, args.seq_length)[1]

    check_linear_gaussian_elbo(obs_seqs, p, theta)
    avg_evidence = jnp.mean(jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, theta))(obs_seqs))
    print('Avg evidence:', avg_evidence)



    q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim, 
                            transition_matrix_conditionning=args.transition_matrix_conditionning) # specify the structure of the true model, but init params are sampled during optimisiation     

    trainer = SVITrainer(p, q, args.optimizer, args.learning_rate, args.num_epochs, args.batch_size)


    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(key_phi, obs_seqs, theta, args.num_fits, store_every=args.store_every) # returns the best fit (based on the last value of the elbo)
    utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True)
    utils.save_params(params, f'phi_every_{args.store_every}_epochs', save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    args = argparse.Namespace()

    experiment_name = 'linear_model'
    save_dir = os.path.join(os.path.join('experiments', experiment_name))
    os.mkdir(save_dir)


    args.seed_theta = 1326
    args.seed_phi = 4569

    args.state_dim, args.obs_dim = 1,1 
    args.transition_matrix_conditionning = 'diagonal'

    args.seq_length = 64
    args.num_seqs = 4096

    args.optimizer = 'adam'
    args.batch_size = 64
    args.learning_rate = 1e-2
    args.num_epochs = 150
    args.store_every = args.num_epochs // 5
    args.num_fits = 5

    utils.save_args(args, 'train_args', save_dir)
    main(args, save_dir)

