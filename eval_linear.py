import jax 
import jax.numpy as jnp
from jax import config 

config.update("jax_enable_x64",True)

import backward_ica.hmm as hmm
import backward_ica.utils as utils
import matplotlib.pyplot as plt
import os
# utils.enable_x64(True)

def main(train_args, eval_args):

    utils.set_global_cov_mode(train_args)
    
    key_eval = jax.random.PRNGKey(eval_args.seed)

    p = hmm.LinearGaussianHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                            transition_bias=train_args.transition_bias,
                            emission_bias=train_args.emission_bias) # specify the structure of the true model

    theta_star = utils.load_params('theta', train_args.save_dir)

    # key, subkey = jax.random.split(key_mle, 2)
    # key, subkey = jax.random.split(key, 2)
    # obs_seqs = p.sample_multiple_sequences(subkey, theta_star, train_args.num_seqs, train_args.seq_length)[1]

    # print('Computing evidence...')
    # avg_evidence_rmle = jnp.mean(jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, theta_star))(obs_seqs))
    # print('Avg evidence:', avg_evidence_rmle)

    # theta_mle, logls_mle = p.fit_kalman_rmle(key, 
    #                                         obs_seqs, 
    #                                         train_args.optimizer, 
    #                                         train_args.learning_rate, 
    #                                         train_args.batch_size, 
    #                                         train_args.num_epochs)

    # print('Computing evidence...')
    # avg_evidence_rmle = jnp.mean(jax.vmap(lambda obs_seq: p.likelihood_seq(obs_seq, theta_mle))(obs_seqs))
    # print('Avg evidence fitted RMLE:', avg_evidence_rmle)

    (best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence) = utils.load_train_logs(train_args.save_dir)

    # plt.axhline(y=avg_evidence_rmle, c='black', label = '$log p_{\\theta}(x)$', linestyle='dotted')
    # plt.plot(logls_mle, label='rmle')
    # plt.savefig(os.path.join(eval_args.save_dir,'rmle'))
    # plt.clf()

    utils.plot_training_curves(best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence, eval_args.save_dir)
    
    state_seqs, obs_seqs = p.sample_multiple_sequences(key_eval, theta_star, eval_args.num_seqs, eval_args.seq_length)
    timesteps = range(1, eval_args.seq_length, eval_args.step)

    smoothing_theta_star = utils.multiple_length_linear_backward_smoothing(obs_seqs, p, theta_star, timesteps)

    
    q = hmm.LinearGaussianHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                            transition_bias=train_args.transition_bias,
                            emission_bias=train_args.emission_bias)

    phi_at_multiple_epochs = utils.load_params(f'phi', train_args.save_dir)
    best_epoch = int(stored_epoch_nbs[1])

    best_params = phi_at_multiple_epochs[1]
    phi_at_multiple_epochs = {epoch_nb: phi_at_multiple_epochs[0][idx] for idx, epoch_nb in enumerate(stored_epoch_nbs[0][0])}
    
    phi_at_multiple_epochs = {k:v for k,v in phi_at_multiple_epochs.items() if k > 40}
    # phi_at_multiple_epochs = dict()
    # phi_at_multiple_epochs[best_epoch] = best_params


    smoothing_q_phi = dict()
    for epoch_nb, phi in phi_at_multiple_epochs.items():
        smoothing_q_phi[f'Variational {epoch_nb}'] = utils.multiple_length_linear_backward_smoothing(obs_seqs, q, phi, timesteps)

    # smoothing_q_phi['MLE'] = utils.multiple_length_linear_backward_smoothing(obs_seqs, p, theta_mle, timesteps)

    utils.plot_multiple_length_smoothing(state_seqs, smoothing_theta_star, smoothing_q_phi, timesteps, 'true', 'variational', eval_args.save_dir)



if __name__ == '__main__':
    import os 
    import argparse

    eval_args = argparse.Namespace()

    train_path = 'experiments/p_linear/trainings/q_linear/2022_05_17__19_32_42'
    train_path_splitted = train_path.split('/')
    eval_name = f'{train_path_splitted[-2]}_{train_path_splitted[-1]}'

    eval_args.save_dir = os.path.join(os.path.join('experiments', 'p_linear', 'evals', eval_name))
    os.makedirs(eval_args.save_dir, exist_ok=True)
    train_args = utils.load_args('train_args', train_path)

    eval_args.num_seqs = 5
    eval_args.seq_length = 2000
    eval_args.step = 100
    eval_args.seed = 0

    utils.save_args(eval_args, 'eval_args', eval_args.save_dir)

    main(train_args, eval_args)


