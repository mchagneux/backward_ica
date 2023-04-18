import jax 
import jax.numpy as jnp
from src import variational

import src.utils.misc as misc
import src.stats.hmm as hmm
import src.variational as variational
import src.stats as stats
from src.training import SVITrainer, define_frozen_tree
jax.config.update('jax_disable_jit', False)
jax.config.update('jax_platform_name', 'cpu')

    
def main(args):


    if args.float64: 
        misc.enable_x64(True)
    stats.set_parametrization(args)

    p = hmm.get_generative_model(misc.load_args('args', args.exp_dir))
    theta_star = misc.load_params('theta_star', args.exp_dir)
    data = jnp.load(os.path.join(args.exp_dir, 'obs_seqs.npy'))


    key_phi = jax.random.PRNGKey(args.seed)

    
    q = variational.get_variational_model(args)


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
                        seq_length=data.shape[1],
                        online_mode=args.online_mode)


    key_params, key_batcher, key_montecarlo = jax.random.split(key_phi, 3)

    print('Online mode:', args.online_mode)

    params = trainer.multi_fit(key_params, key_batcher, key_montecarlo, 
                                                            data=data, 
                                                            num_fits=args.num_fits,
                                                            log_dir=args.save_dir,
                                                            store_every=args.store_every,
                                                            args=args)[0] # returns the best fit (based on the last value of the elbo)
    
    # utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), args.save_dir, plot=True, best_epochs_only=True)
    misc.save_params(params, 'phi', args.save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    from datetime import datetime
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--exp_dir', type=str, default='experiments/p_linear/2023_01_16__16_58_54')

    parser.add_argument('--online_mode', type=str, default='off')
    parser.add_argument('--num_fits', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--num_samples', type=int, default=10)
    
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--store_every', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    # args.online = True
    args.full_mc = '_mc' in args.model # whether to force the use the full MCMC ELBO (e.g. prevent using closed-form terms even with linear models)
    args.frozen_params  = args.model.split('freeze__')[1:] # list of parameter groups which are not learnt
    args.save_dir = os.path.join(args.exp_dir, args.model)
    os.makedirs(args.save_dir, exist_ok=True)

    if len(args.model.split('__')) > 1:
        args.model_options = args.model.split('__')[1]
    else: 
        args.model_options = ''

    args.model = args.model.split('__')[0]


    args = misc.get_defaults(args)
    args.transition_matrix_conditionning = 'diagonal'
    args.range_transition_map_params = [-1,1]
    args.transition_bias = False
        
    misc.save_args(args, 'args', args.save_dir)
    args_p = misc.load_args('args', args.exp_dir)
    args.state_dim, args.obs_dim = args_p.state_dim, args_p.obs_dim


    main(args)

