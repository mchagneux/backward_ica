import jax
import jax.numpy as jnp 
import os 
import backward_ica.utils as utils
import backward_ica.stats.hmm as hmm
import backward_ica.stats as stats
from backward_ica.elbos import check_linear_gaussian_elbo

def main(args):
    
    if args.float64: 
        utils.enable_x64(True)

    stats.set_parametrization(args)

    key_theta = jax.random.PRNGKey(args.seed)
    key_params, key_gen, key_smc = jax.random.split(key_theta, 3)
    p, theta_star = hmm.get_generative_model(args, 
                                            key_for_random_params=key_params)

    utils.save_params(theta_star, 'theta_star', args.exp_dir)

    state_seqs, obs_seqs = p.sample_multiple_sequences(key_gen, 
                                            theta_star, 
                                            args.num_seqs, 
                                            args.seq_length, 
                                            single_split_seq=args.single_split_seq,
                                            load_from=args.load_from,
                                            loaded_seq=args.loaded_seq)

    jnp.save(os.path.join(args.exp_dir, 'state_seqs.npy'), state_seqs)
    jnp.save(os.path.join(args.exp_dir, 'obs_seqs.npy'), obs_seqs)



    if args.compute_oracle_evidence:
        print('Computing evidence...')

        evidence_keys = jax.random.split(key_smc, args.num_seqs)

        if args.model == 'linear':
            evidence_func = lambda key, obs_seq, params: p.likelihood_seq(obs_seq, params)
            check_linear_gaussian_elbo(p, args.num_seqs, args.seq_length)

        else: 
            evidence_func = p.likelihood_seq

        avg_evidence = jnp.mean(jax.vmap(jax.jit(lambda key, obs_seq: evidence_func(key, obs_seq, theta_star)))(evidence_keys, obs_seqs)) / args.seq_length

        print('Oracle evidence:', avg_evidence)


        

if __name__ == '__main__':

    import argparse 
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='chaotic_rnn')
    parser.add_argument('--dims', type=int, nargs='+', default=(5,5))
    parser.add_argument('--num_seqs', type=int, default=1000)
    parser.add_argument('--seq_length',type=int, default=2000)
    parser.add_argument('--single_split_seq', type=bool, default=False)
    parser.add_argument('--load_from', type=str, default='')
    parser.add_argument('--transition_bias', type=bool, default=False)
    parser.add_argument('--emission_bias', type=bool, default=False)
    parser.add_argument('--compute_oracle_evidence',type=bool, default=True)
    parser.add_argument('--exp_dir', type=str, default='experiments/tests')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--loaded_seq', action='store_true', default=False)

    args = parser.parse_args()
    args.state_dim, args.obs_dim = args.dims
    del args.dims
    os.makedirs(args.exp_dir, exist_ok=True)
    args = utils.get_defaults(args)
    utils.save_args(args, 'args', args.exp_dir)
    main(args)



