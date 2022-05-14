import jax 
import jax.numpy as jnp
from jax import config 

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils




def main(dirs, eval_args, save_dir):


    train_logs = [utils.load_train_logs(train_dir) for train_dir in dirs]
    
    utils.superpose_training_curves(train_logs, eval_args.methods, save_dir, start_index=0)
    key = jax.random.PRNGKey(eval_args.seed)

    args_V1 = utils.load_args('train_args', dir_VI_1)


    p = hmm.NonLinearGaussianHMM(state_dim=args_V1.state_dim, 
                            obs_dim=args_V1.obs_dim, 
                            transition_matrix_conditionning=args_V1.transition_matrix_conditionning,
                            layers=args_V1.emission_map_layers,
                            slope=args_V1.slope,
                            num_particles=eval_args.num_particles) # specify the structure of the true model

    theta = utils.load_params('theta', dir_VI_1)

    key_gen, key_smc = jax.random.split(key,2)
    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta, eval_args.num_seqs, eval_args.seq_length)
    timesteps = range(1, eval_args.seq_length, eval_args.step)
    



    
    q = hmm.LinearGaussianHMM(state_dim=args_V1.state_dim, 
                            obs_dim=args_V1.obs_dim, 
                            transition_matrix_conditionning=args_V1.transition_matrix_conditionning)

    phi = utils.load_params('phi', dir_VI_1)


    smoothing_q_phi_1 = utils.multiple_length_linear_backward_smoothing(obs_seqs, q, phi, timesteps)

    args_V2 = utils.load_args('train_args',dir_VI_2)
    q = hmm.NeuralLinearBackwardSmoother(args_V2.state_dim, args_V2.obs_dim, update_layers=args_V2.update_layers)

    phi = utils.load_params('phi', dir_VI_2)


    smoothing_q_phi_2 = utils.multiple_length_linear_backward_smoothing(obs_seqs, q, phi, timesteps)

    smoothing_q_phi = {'linearVI':smoothing_q_phi_1, 'nonlinearVI':smoothing_q_phi_2}
    utils.compare_multiple_length_smoothing(state_seqs, smoothing_p_theta, smoothing_q_phi, timesteps, f'ffbsi_{eval_args.num_particles}', 'VI', save_dir)


if __name__ == '__main__':
    import os 
    import argparse

    from datetime import datetime 
    eval_args = argparse.Namespace()
    eval_args.methods = ['p_noninjective_q_linear_2022_05_14__11_33_05', 
                        'p_noninjective_q_nonlinear_johnson_2022_05_14__11_33_15',
                        'p_noninjective_q_nonlinear_ours_2022_05_14__11_33_25']
                        
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    save_dir = os.path.join('experiments', f'compararison_{date}')
    os.mkdir(save_dir)


    eval_args.num_seqs = 5
    eval_args.seq_length = 500
    eval_args.num_particles = 1000
    eval_args.step = 50
    eval_args.seed = 0

    dirs = [os.path.join('experiments/p_nonlinear', method_name) for method_name in eval_args.methods]

    utils.save_args(eval_args, 'eval_args', save_dir)

    main(dirs, eval_args, save_dir)


