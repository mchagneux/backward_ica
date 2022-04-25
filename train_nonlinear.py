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

    state_seqs, obs_seqs = hmm.sample_multiple_sequences(key_gen, p.sample_seq, theta, args.num_seqs, args.seq_length)

    import matplotlib.pyplot as plt 
    test_points = jnp.linspace(state_seqs.min(),state_seqs.max(), 100).reshape(-1, args.state_dim)
    plt.plot(test_points, p.emission_kernel.map(test_points, theta.emission))
    plt.show()

    smc_keys = jax.random.split(key_smc, args.num_seqs)

    avg_evidence = jnp.mean(jax.vmap(jax.jit(lambda obs_seq, key: p.likelihood_seq(obs_seq, 
                                                                        theta, 
                                                                        key,
                                                                        args.num_particles)))(obs_seqs, smc_keys))


    print('Avg evidence:', avg_evidence)

    # q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
    #                         obs_dim=args.obs_dim,
    #                         transition_matrix_conditionning=args.transition_matrix_conditionning)


    # q = hmm.LinearGaussianHMM(state_dim=args.state_dim, 
    #                             obs_dim=args.obs_dim,
    #                             transition_matrix_conditionning=args.transition_matrix_conditionning)

    q = hmm.NeuralBackwardSmoother(state_dim=args.state_dim, obs_dim=args.obs_dim)

    trainer = SVITrainer(p, q, args.optimizer, args.learning_rate, args.num_epochs, args.batch_size, args.num_samples)
    phi, avg_elbos = trainer.multi_fit(key_phi, obs_seqs, theta, args.num_fits) # returns the best fit (based on the last value of the elbo)
    utils.plot_training_curves(avg_elbos, save_dir, avg_evidence)

if __name__ == '__main__':

    import argparse
    import os 
    args = argparse.Namespace()

    experiment_name = 'nonlinear_model_nonlinear_var_3'
    save_dir = os.path.join(os.path.join('experiments', experiment_name))
    os.mkdir(save_dir)


    args.seed_theta = 1327
    args.seed_phi = 4569

    args.state_dim, args.obs_dim = 1,1 
    args.transition_matrix_conditionning = 'diagonal'
    args.hidden_layer_sizes = (4,4,4,4)
    args.slope = 0.8

    args.seq_length = 16
    args.num_seqs = 6400

    args.optimizer = 'adam'
    args.batch_size = 64
    args.learning_rate = 1e-2
    args.num_epochs = 300
    args.store_every = args.num_epochs // 5
    args.num_fits = 5
    

    args.num_particles = 1000
    args.num_samples = 10


    utils.save_args(args, 'train_args', save_dir)
    main(args, save_dir)