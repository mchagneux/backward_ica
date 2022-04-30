import jax 
import jax.numpy as jnp
import optax 
from jax import config 
import pickle

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils
from backward_ica.svi import SVITrainer


def main(args, save_dir):


    key_theta = jax.random.PRNGKey(args.seed_theta)
    key_phi = jax.random.PRNGKey(args.seed_phi)

    p = hmm.NonLinearGaussianHMM(state_dim=args.state_dim, 
                            obs_dim=args.obs_dim, 
                            transition_matrix_conditionning=args.transition_matrix_conditionning,
                            hidden_layer_sizes=args.hidden_layer_sizes,
                            slope=args.slope) # specify the structure of the true model

    key_params, key_gen, key_smc = jax.random.split(key_theta, 3)

    theta = p.get_random_params(key_params) # sample params randomly (but covariances are fixed to default values)
    utils.save_params(theta, 'theta', save_dir)

    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta, args.num_seqs, args.seq_length)

    import matplotlib.pyplot as plt 

    test_points = jnp.linspace(state_seqs.min(),state_seqs.max(), 100).reshape(-1, args.state_dim)
    plt.plot(test_points, p.emission_kernel.map(test_points, theta.emission))
    plt.savefig(os.path.join(save_dir,'emission_map_on_states_support'))
    plt.clf()

    plt.scatter(range(len(state_seqs[0])), state_seqs[0])
    plt.savefig(os.path.join(save_dir,'example_states'))
    plt.clf()

    smc_keys = jax.random.split(key_smc, args.num_seqs)

    avg_evidence = jnp.mean(jax.vmap(jax.jit(lambda obs_seq, key: p.likelihood_seq(obs_seq, 
                                                                        theta, 
                                                                        key,
                                                                        args.num_particles)))(obs_seqs, smc_keys))


    print('Avg evidence:', avg_evidence)


    q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
                                obs_dim=args.obs_dim,
                                transition_matrix_conditionning=args.transition_matrix_conditionning)

    # q = hmm.NeuralBackwardSmoother(state_dim=args.state_dim, obs_dim=args.obs_dim)

    trainer = SVITrainer(p, q, args.optimizer, args.learning_rate, args.num_epochs, args.batch_size, args.num_samples)

    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(*jax.random.split(key_phi,3), obs_seqs, theta, args.num_fits) # returns the best fit (based on the last value of the elbo)
    utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True)
    utils.save_params(params, f'phi_every_{args.store_every}_epochs', save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    import sys

    args = argparse.Namespace()

    experiment_name = 'nonlinear_refactor'
    save_dir = os.path.join(os.path.join('experiments', experiment_name))
    
    os.mkdir(save_dir)

    # sys.stdout = open(os.path.join(save_dir, 'train_logs.txt'), 'w')

    args.seed_theta = 1329
    args.seed_phi = 4569

    args.state_dim, args.obs_dim = 1,1 
    args.transition_matrix_conditionning = 'diagonal'
    args.hidden_layer_sizes = ()
    args.slope = 0

    args.seq_length = 16
    args.num_seqs = 6400

    args.optimizer = 'adam'
    args.batch_size = 64
    args.learning_rate = 1e-2
    args.num_epochs = 150
    args.store_every = args.num_epochs // 5
    args.num_fits = 5
    

    args.num_particles = 1000
    args.num_samples = 1

    # os.environ["XLA_FLAGS"] = f'--xla_force_host_platform_device_count={args.batch_size}'


    utils.save_args(args, 'train_args', save_dir)
    main(args, save_dir)
    # sys.stdout.close()
