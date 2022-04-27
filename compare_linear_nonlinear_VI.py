import jax 
import jax.numpy as jnp
from jax import config 

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils




def main(dir_VI_1, dir_VI_2, eval_args, save_dir):

    key = jax.random.PRNGKey(eval_args.seed)

    args_V1 = utils.load_args('train_args', dir_VI_1)

    p = hmm.NonLinearGaussianHMM(state_dim=args_V1.state_dim, 
                            obs_dim=args_V1.obs_dim, 
                            transition_matrix_conditionning=args_V1.transition_matrix_conditionning,
                            hidden_layer_sizes=args_V1.hidden_layer_sizes,
                            slope=args_V1.slope) # specify the structure of the true model

    theta = utils.load_params('theta', dir_VI_1)

    key_gen, key_smc = jax.random.split(key,2)
    state_seqs, obs_seqs = hmm.sample_multiple_sequences(key_gen, p.sample_seq, theta, eval_args.num_seqs, eval_args.seq_length)
    timesteps = range(1, eval_args.seq_length, eval_args.step)

    smoothing_p_theta = utils.multiple_length_ffbsi_smoothing(obs_seqs, p, theta, timesteps, key_smc, eval_args.num_particles)

    training_curves_1 = utils.load_train_logs(dir_VI_1)
    training_curves_2 = utils.load_train_logs(dir_VI_2)

    utils.superpose_training_curves(training_curves_1, training_curves_2, 'linearVI', 'nonlinearVI', save_dir)
    
    q = hmm.LinearGaussianHMM(state_dim=args_V1.state_dim, 
                            obs_dim=args_V1.obs_dim, 
                            transition_matrix_conditionning=args_V1.transition_matrix_conditionning)

    phi = utils.load_params(f'phi_every_{args_V1.store_every}_epochs', dir_VI_1)[args_V1.num_epochs-1]


    smoothing_q_phi_1 = utils.multiple_length_linear_backward_smoothing(obs_seqs, q, phi, timesteps)

    args_V2 = utils.load_args('train_args',dir_VI_2)
    q = hmm.NeuralBackwardSmoother(args_V2.state_dim, args_V2.obs_dim)

    phi = utils.load_params(f'phi_every_{args_V2.store_every}_epochs', dir_VI_2)[args_V2.num_epochs-1]


    smoothing_q_phi_2 = utils.multiple_length_linear_backward_smoothing(obs_seqs, q, phi, timesteps)

    smoothing_q_phi = {'linearVI':smoothing_q_phi_1, 'nonlinearVI':smoothing_q_phi_2}
    utils.compare_multiple_length_smoothing(state_seqs, smoothing_p_theta, smoothing_q_phi, timesteps, f'ffbsi_{eval_args.num_particles}', 'VI', save_dir)


if __name__ == '__main__':
    import os 
    import argparse

    eval_args = argparse.Namespace()

    dir_VI_1 = os.path.join('experiments', 'nonlinear_p_theta_linear_q_phi')
    dir_VI_2 = os.path.join('experiments', 'nonlinear_p_theta_nonlinear_q_phi')

    save_dir = os.path.join('experiments', 'compare_linear_nonlinear_VI')
    os.mkdir(save_dir)

    eval_args.num_seqs = 5
    eval_args.seq_length = 500
    eval_args.num_particles = 1000
    eval_args.step = 50
    eval_args.seed = 0

    utils.save_args(eval_args, 'eval_args', save_dir)

    main(dir_VI_1, dir_VI_2, eval_args, save_dir)


