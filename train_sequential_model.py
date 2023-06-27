import jax 
import jax.numpy as jnp
from src import variational

import src.utils.misc as misc
import src.stats.hmm as hmm
import src.variational as variational
import src.stats as stats
from src.training import SVITrainer, define_frozen_tree
jax.config.update('jax_disable_jit', False)
# jax.config.update('jax_platform_name', 'cpu')
import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

    
def main(args):


    if args.float64: 
        misc.enable_x64(True)
    stats.set_parametrization(args)

    p = hmm.get_generative_model(misc.load_args('args', args.exp_dir))
    theta_star = misc.load_params('theta_star', args.exp_dir)
    ys = jnp.load(os.path.join(args.exp_dir, 'obs_seqs.npy'))
    xs = jnp.load(os.path.join(args.exp_dir, 'state_seqs.npy'))



    key_phi = jax.random.PRNGKey(args.seed)

    
    q = variational.get_variational_model(args)


    # q.transition_kernel = p.transition_kernel
    frozen_params = define_frozen_tree(key_phi, 
                                        args.frozen_params, 
                                        q, 
                                        theta_star)



    
    trainer = SVITrainer(p=p, 
                        theta_star=theta_star,
                        q=q, 
                        optimizer=args.optimizer, 
                        learning_rate=args.learning_rate,
                        optim_options=args.optim_options,
                        num_epochs=args.num_epochs, 
                        batch_size=args.batch_size, 
                        num_samples=args.num_samples,
                        force_full_mc=args.full_mc,
                        frozen_params=frozen_params,
                        seq_length=ys.shape[1],
                        training_mode=args.training_mode,
                        elbo_mode=args.elbo_mode)



    print('Elbo mode:', args.elbo_mode)

    best_params_across_all_fits, best_params_per_fit = trainer.multi_fit(key_phi,
                                                            data=(xs,ys), 
                                                            num_fits=args.num_fits,
                                                            log_dir=args.save_dir,
                                                            args=args) # returns the best fit (based on the last value of the elbo)
    
    misc.save_params(best_params_across_all_fits, 'phi', args.save_dir)

    # utils.save_train_logs((best_fit_idx, stored_epoch_nbs, avg_elbos, avg_evidence), args.save_dir, plot=True, best_epochs_only=True)
    for fit_nb in range(args.num_fits):
        misc.save_params(best_params_per_fit[fit_nb], f'phi_fit_{fit_nb}', args.save_dir)

if __name__ == '__main__':

    import argparse
    import os 
    from datetime import datetime
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=str, default='neural_backward,10.50.adam,1e-3,cst.true_online,500.score,truncated,paris,variance_reduction,bptt_depth_1')
    parser.add_argument('--exp_dir', type=str, default='experiments/p_chaotic_rnn/2023_06_26__08_00_15')
    parser.add_argument('--num_fits', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--store_every', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    # args.online = True
    args.full_mc = '_mc' in args.settings # whether to force the use the full MCMC ELBO (e.g. prevent using closed-form terms even with linear models)
    args.frozen_params  = '' #args.settings.split('freeze__')[1:] # list of parameter groups which are not learnt
    args.save_dir = os.path.join(args.exp_dir, args.settings)
    os.makedirs(args.save_dir, exist_ok=True)
    
    args.model, num_samples, optim, args.training_mode, args.elbo_mode = args.settings.split('.')
    args.num_samples = int(num_samples)
    args.optimizer, learning_rate, args.optim_options = optim.split(',')
    args.learning_rate = float(learning_rate)
    args = misc.get_defaults(args)

        
    misc.save_args(args, 'args', args.save_dir)
    args_p = misc.load_args('args', args.exp_dir)
    args.state_dim, args.obs_dim = args_p.state_dim, args_p.obs_dim


    main(args)

