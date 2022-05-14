import jax 
import jax.numpy as jnp
from jax import config 

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils
import os 
import pickle



def main(train_args, eval_args):

    key = jax.random.PRNGKey(eval_args.seed)

    p = hmm.NonLinearGaussianHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                            layers=train_args.emission_map_layers,
                            slope=train_args.slope) # specify the structure of the true model

    theta = utils.load_params('theta', train_args.save_dir)

    key_gen, _ = jax.random.split(key,2)
    obs_seqs = p.sample_multiple_sequences(key_gen, theta, eval_args.num_seqs, eval_args.seq_length)[1]
    timesteps = range(1, eval_args.seq_length, eval_args.step)


    if train_args.q_version == 'linear':

        q = hmm.LinearGaussianHMM(state_dim=train_args.state_dim, 
                                obs_dim=train_args.obs_dim,
                                transition_matrix_conditionning=train_args.transition_matrix_conditionning)

    else: 
        version = train_args.q_version.split('_')[1]
        q = hmm.NeuralLinearBackwardSmoother(state_dim=train_args.state_dim, 
                                        obs_dim=train_args.obs_dim, 
                                        use_johnson=(version == 'johnson'),
                                        update_layers=train_args.update_layers)


    phi = utils.load_params('phi', train_args.save_dir)

    smoothing_q_phi = utils.multiple_length_linear_backward_smoothing(obs_seqs, q, phi, timesteps)
    with open(os.path.join(eval_args.save_dir, 'smoothing_results'), 'wb') as f:
        pickle.dump(smoothing_q_phi, f)



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--save_dir',type=str)
    eval_args = parser.parse_args()

    eval_args.num_seqs = 5
    eval_args.seq_length = 200
    eval_args.num_particles = 1000
    eval_args.step = 10
    eval_args.seed = 0
    train_args = utils.load_args('train_args', eval_args.train_dir)

    utils.save_args(eval_args, 'eval_args', eval_args.save_dir)

    main(train_args, eval_args)


