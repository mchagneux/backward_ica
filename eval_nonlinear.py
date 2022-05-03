import jax 
import jax.numpy as jnp
from jax import config 

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils




def main(train_args, eval_args, save_dir):

    key = jax.random.PRNGKey(eval_args.seed)

    p = hmm.NonLinearGaussianHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                            layers=train_args.layers,
                            slope=train_args.slope) # specify the structure of the true model

    theta = utils.load_params('theta', save_dir)

    key_gen, key_smc = jax.random.split(key,2)
    state_seqs, obs_seqs = hmm.sample_multiple_sequences(key_gen, p.sample_seq, theta, eval_args.num_seqs, eval_args.seq_length)
    timesteps = range(1, eval_args.seq_length, eval_args.step)

    smoothing_p_theta = utils.multiple_length_ffbsi_smoothing(obs_seqs, p, theta, timesteps, key_smc, eval_args.num_particles)


    q = hmm.LinearGaussianHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning)

    phi = utils.load_params(f'phi_every_{train_args.store_every}_epochs', save_dir)[train_args.num_epochs-1]

    # smoothing_q_phi = dict()
    # for epoch_nb, phi in phi_at_multiple_epochs.items():
    smoothing_q_phi = utils.multiple_length_linear_backward_smoothing(obs_seqs, q, phi, timesteps)

    utils.plot_multiple_length_smoothing(state_seqs, smoothing_p_theta, smoothing_q_phi, timesteps, f'ffbsi_{eval_args.num_particles}', 'linearVI', save_dir)



if __name__ == '__main__':
    import os 
    import argparse

    eval_args = argparse.Namespace()

    experiment_name = 'nonlinear_model_linear_var'
    save_dir = os.path.join(os.path.join('experiments', experiment_name))
    train_args = utils.load_args('train_args',save_dir)

    eval_args.num_seqs = 5
    eval_args.seq_length = 200
    eval_args.num_particles = 1000
    eval_args.step = 10
    eval_args.seed = 0

    utils.save_args(eval_args, 'eval_args', save_dir)

    main(train_args, eval_args, save_dir)


