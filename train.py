import jax 
import jax.numpy as jnp

import backward_ica.utils as utils
import backward_ica.stats.hmm as hmm
import backward_ica.variational.models as variational_models
import backward_ica.stats as stats
from backward_ica.training import SVITrainer, define_frozen_tree

utils.enable_x64(True)

def main(args, save_dir):

    stats.set_parametrization(args)

    key_theta = jax.random.PRNGKey(args.seed_theta)
    key_phi = jax.random.PRNGKey(args.seed_phi)

    key_params, key_gen, key_smc = jax.random.split(key_theta, 3)

    p, theta_star = hmm.get_generative_model(args, key_for_random_params=key_params)

    utils.save_params(theta_star, 'theta', save_dir)

    obs_seqs = p.sample_multiple_sequences(key_gen, 
                                            theta_star, 
                                            args.num_seqs, 
                                            args.seq_length, 
                                            single_split_seq=args.single_split_seq,
                                            loaded_data=args.loaded_data)[1]


    evidence_keys = jax.random.split(key_smc, args.num_seqs)

    if args.compute_oracle_evidence:
        print('Computing evidence...')

        avg_evidence = jnp.mean(jax.vmap(jax.jit(lambda key, obs_seq: p.likelihood_seq(key, obs_seq, theta_star)))(evidence_keys, obs_seqs)) / args.seq_length

        print('Oracle evidence:', avg_evidence)

    q = variational_models.get_variational_model(args, p=p)


    frozen_params = define_frozen_tree(key_phi, 
                                        args.frozen_params, 
                                        q, 
                                        theta_star)
    

    trainer = SVITrainer(p=p, 
                        theta_star=theta_star,
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        num_samples=args.num_samples,
                        force_full_mc=args.full_mc,
                        frozen_params=frozen_params,
                        online=args.online,
                        sweep_sequence=False)


    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi, 3)

    params, (best_fit_idx, stored_epoch_nbs, avg_elbos) = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                                                            data=obs_seqs, 
                                                            num_fits=args.num_fits,
                                                            log_dir=save_dir,
                                                            args=args) # returns the best fit (based on the last value of the elbo)
    
    # utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), save_dir, plot=True, best_epochs_only=True)
    utils.save_params(params, 'phi', save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    from datetime import datetime

    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


    parser = argparse.ArgumentParser()
    parser.add_argument('--p_version', type=str, default='chaotic_rnn')
    parser.add_argument('--q_version', type=str, default='johnson_forward')

    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--args_path', type=str, default='')
    parser.add_argument('--dims', type=int, nargs='+', default=(5,5))
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--num_seqs', type=int, default=1000)
    parser.add_argument('--seq_length',type=int, default=100)
    parser.add_argument('--compute_oracle_evidence',type=bool, default=False)
    parser.add_argument('--load_sequences',type=bool, default=False)

    args = parser.parse_args()

    save_dir = args.save_dir

    if save_dir=='':
        save_dir = os.path.join('experiments','tests',date)
        os.makedirs(save_dir, exist_ok=True)

    if args.args_path != '':
        args = utils.load_args('train_args',args.args_path)
        args.save_dir = save_dir
    else:
        args = utils.get_config(external_args=args)
    utils.save_args(args, 'train_args', save_dir)

    main(args, save_dir)
