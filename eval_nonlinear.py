import jax 
import jax.numpy as jnp
from jax import config 

config.update("jax_enable_x64", True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils
import os 
import dill



def main(train_args, eval_args):

    key = jax.random.PRNGKey(eval_args.seed)
    
    utils.set_defaults(train_args)

    p = hmm.NonLinearHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                            range_transition_map_params=train_args.range_transition_map_params,
                            layers=train_args.emission_map_layers,
                            slope=train_args.slope,
                            transition_bias=train_args.transition_bias,
                            injective=train_args.injective) # specify the structure of the true model

    theta = utils.load_params('theta', train_args.save_dir)

    key_gen, _ = jax.random.split(key,2)
    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, theta, eval_args.num_seqs, eval_args.seq_length)
    timesteps = range(2, eval_args.seq_length, eval_args.step)

    print(obs_seqs[0][list(timesteps)[0]])

    if train_args.q_version == 'linear':

        q = hmm.LinearGaussianHMM(state_dim=train_args.state_dim, 
                                obs_dim=train_args.obs_dim,
                                transition_matrix_conditionning=train_args.transition_matrix_conditionning, 
                                range_transition_map_params=train_args.range_transition_map_params,
                                transition_bias=train_args.transition_bias,
                                emission_bias=False)

    else: 
        version = train_args.q_version.split('_')[1]
        q = hmm.NeuralLinearBackwardSmoother(state_dim=train_args.state_dim, 
                                        obs_dim=train_args.obs_dim, 
                                        use_johnson=(version == 'johnson'),
                                        range_transition_map_params=train_args.range_transition_map_params,
                                        update_layers=train_args.update_layers,
                                        transition_bias=train_args.transition_bias)


    phi = utils.load_params('phi', train_args.save_dir)

    smoothing_q_phi = utils.multiple_length_linear_backward_smoothing(obs_seqs, q, phi, timesteps)
    with open(os.path.join(eval_args.save_dir, 'smoothing_results'), 'wb') as f:
        dill.dump(smoothing_q_phi, f)



if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_dir', type=str)
    parser.add_argument('--save_dir',type=str)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--step', type=int)
    parser.add_argument('--num_seqs', type=int)
    parser.add_argument('--seq_length', type=int)

    eval_args = parser.parse_args()
    train_args = utils.load_args('train_args', eval_args.train_dir)

    utils.save_args(eval_args, 'eval_args', eval_args.save_dir)

    main(train_args, eval_args)


