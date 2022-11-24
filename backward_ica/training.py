from backward_ica.offline_elbos import *
from backward_ica.online_smoothing import *
from backward_ica.stats.hmm import * 
from backward_ica.variational.models import *

import tensorflow as tf 
from jax.tree_util import tree_flatten
import jax
from jax import vmap, value_and_grad, numpy as jnp

import optax 

def winsorize_grads():
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        flattened_updates = jnp.concatenate([arr.flatten() for arr in tree_flatten(updates)[0]])
        high_value = jnp.sort(jnp.abs(flattened_updates))[int(0.90*flattened_updates.shape[0])]
        return tree_map(lambda x: jnp.clip(x, -high_value, high_value), updates), ()
    return optax.GradientTransformation(init_fn, update_fn)

def define_frozen_tree(key, frozen_params, q, theta_star):

    # key_theta, key_phi = random.split(key, 2)

    frozen_phi = q.get_random_params(key)
    frozen_phi = tree_map(lambda x: '', frozen_phi)


    # if 'theta' in frozen_params: 
    #     frozen_theta = theta_star 

    if 'prior_phi' in frozen_params:
        if isinstance(q, LinearGaussianHMM) or (isinstance(q, NeuralLinearBackwardSmoother) and q.explicit_proposal):
            frozen_phi.prior = theta_star.prior
        # else:
        #     if isinstance(frozen_phi, GeneralBackwardSmootherParams):
        #         frozen_phi.prior = GeneralBackwardSmootherParams(q.get_init_state(), frozen_phi.filt_update, frozen_phi.backwd)
        #     else: 
        #         frozen_phi.prior = q.get_init_state()
    
    if 'transition_phi' in frozen_params:
        # if isinstance(q, NeuralBackwardSmoother):
        #     raise NotImplementedError
        # else: 
            frozen_phi.transition = theta_star.transition

    if 'covariances' in frozen_params: 
        frozen_phi.transition.noise.scale = theta_star.transition.noise.scale
    
    # frozen_params = (frozen_theta, frozen_phi)

    return frozen_phi
    
class SVITrainer:

    def __init__(self, p:HMM, 
                theta_star,
                q:BackwardSmoother, 
                optimizer, 
                learning_rate, 
                num_epochs, 
                batch_size, 
                num_samples=1, 
                force_full_mc=False,
                frozen_params=None,
                online=False,
                sweep_sequences=False):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.q = q 
        self.q.print_num_params()
        self.p = p 
        
        self.theta_star = theta_star
        self.frozen_params = frozen_params

        self.trainable_params = tree_map(lambda x: x == '', self.frozen_params)
        self.fixed_params = tree_map(lambda x: x != '', self.frozen_params)

        base_optimizer = optax.apply_if_finite(optax.masked(getattr(optax, optimizer)(learning_rate), 
                                                            self.trainable_params), 
                                            max_consecutive_errors=10)

        zero_grads_optimizer = optax.masked(optax.set_to_zero(), self.fixed_params)

        self.optimizer = optax.chain(zero_grads_optimizer, base_optimizer)
        self.sweep_sequences = sweep_sequences
        self.online = online
        

        if isinstance(self.q, TwoFilterSmoother):
            self.elbo = GeneralForwardELBO(self.p, self.q, num_samples)
            self.get_montecarlo_keys = get_keys
            self.loss = lambda key, data, compute_up_to, params: -self.elbo(key, data, compute_up_to, self.p.format_params(self.theta_star), q.format_params(params))

        else: 
            if isinstance(self.p, LinearGaussianHMM) and isinstance(self.q, LinearGaussianHMM) and (not force_full_mc):
                self.elbo = LinearGaussianELBO(self.p, self.q)
                self.get_montecarlo_keys = get_dummy_keys
                self.loss = lambda key, data, compute_up_to, params: -self.elbo(data, compute_up_to, self.p.format_params(self.theta_star), q.format_params(params))
            else: 
                self.elbo = GeneralBackwardELBO(self.p, self.q, num_samples)
                self.get_montecarlo_keys = get_keys
                self.loss = lambda key, data, compute_up_to, params: -self.elbo(key, data, compute_up_to, self.p.format_params(self.theta_star), q.format_params(params))
        
        if self.online: 
            self.elbo = OnlineParisELBO(p=self.p, q=self.q, normalizer=None, num_samples=num_samples)
            self.get_montecarlo_keys = get_keys
            self.loss = lambda key, data, compute_up_to, params: -self.elbo.batch_compute(key, data, self.p.format_params(self.theta_star), q.format_params(params))


        if not self.sweep_sequences: 
            def step(carry, x):
                def batch_step(params, opt_state, batch, keys):
                    neg_elbo_values, grads = vmap(value_and_grad(self.loss, argnums=3), in_axes=(0, 0, None, None))(keys, batch, batch.shape[1]-1, params)

                    avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), grads)
                    updates, opt_state = self.optimizer.update(avg_grads, opt_state, params)
                    params = optax.apply_updates(params, updates)

                    return params, \
                        opt_state, \
                        -jnp.mean(neg_elbo_values)

                data, params, opt_state, subkeys_epoch = carry

                batch_start = x
                batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
                batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)

                params, opt_state, avg_elbo_batch = batch_step(params, opt_state, batch_obs_seq, batch_keys)

                return (data, params, opt_state, subkeys_epoch), avg_elbo_batch

        else: 
            def step(carry, x):
                def batch_step(params, opt_state, batch, keys):
                    def timestep_step(carry, x):
                        params, opt_state, batch, keys = carry 
                        timestep = x
                        neg_elbo_values, grads = vmap(value_and_grad(self.loss, argnums=3), 
                                                        in_axes=(0,0,None,None))(keys, batch, timestep, params)

                        avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), grads)
                        updates, opt_state = self.optimizer.update(avg_grads, opt_state, params)
                        params = optax.apply_updates(params, updates)
                        return (params, opt_state, batch, keys), -jnp.mean(neg_elbo_values)

                    (params, opt_state, _, _), neg_elbo_values = jax.lax.scan(timestep_step, 
                                                                                init=(params, opt_state, batch, keys), 
                                                                                xs=self.timesteps(batch.shape[1]))
                    
                    return params, opt_state, jnp.mean(neg_elbo_values)

                data, params, opt_state, subkeys_epoch = carry
                batch_start = x
                batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
                batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)

                params, opt_state, avg_elbo_batch = batch_step(params, opt_state, batch_obs_seq, batch_keys)

                return (data, params, opt_state, subkeys_epoch), avg_elbo_batch

        self.step = step

    def timesteps(self, seq_length):
        return jnp.arange(0, seq_length)

    def fit(self, key_params, key_batcher, key_montecarlo, data, log_writer=None, args=None):


        num_seqs = data.shape[0]

        params = self.q.get_random_params(key_params, args)

        params = tree_map(lambda param, frozen_param: param if frozen_param == '' else frozen_param, 
                        params, 
                        self.frozen_params)

        opt_state = self.optimizer.init(params)
        subkeys = self.get_montecarlo_keys(key_montecarlo, num_seqs, self.num_epochs)

        avg_elbos = []
        all_params = []
        batch_start_indices = jnp.arange(0, num_seqs, self.batch_size)

        t = tqdm(total=self.num_epochs, desc='Epoch')
        for epoch_nb in range(self.num_epochs):
            t.update(1)
            subkeys_epoch = subkeys[epoch_nb]
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            
            data = jax.random.permutation(subkey_batcher, data)
        

            (_ , params, opt_state, _), avg_elbo_batches = jax.lax.scan(f=self.step,  
                                                                        init=(data, params, opt_state, subkeys_epoch), 
                                                                        xs = batch_start_indices)


            avg_elbo_epoch = jnp.mean(avg_elbo_batches)
            t.set_postfix({'Avg ELBO epoch':avg_elbo_epoch})

            if log_writer is not None:
                with log_writer.as_default():
                    tf.summary.scalar('Epoch ELBO', avg_elbo_epoch, epoch_nb)
                    for batch_nb, avg_elbo_batch in enumerate(avg_elbo_batches):
                        tf.summary.scalar('Minibatch ELBO', avg_elbo_batch, epoch_nb*len(batch_start_indices) + batch_nb)

            avg_elbos.append(avg_elbo_epoch)
            all_params.append(params)
        t.close()
                    
        return all_params, avg_elbos

    def multi_fit(self, key_params, key_batcher, key_montecarlo, data, num_fits, store_every=None, log_dir='', args=None):


        all_avg_elbos = []
        all_params = []
        best_elbos = []
        best_epochs = []
        
        print('Starting training...')
        
        tensorboard_subdir = os.path.join(log_dir, 'tensorboard_logs')
        os.makedirs(tensorboard_subdir, exist_ok=True)
        for fit_nb, subkey_params in enumerate(jax.random.split(key_params, num_fits)):
            log_writer = tf.summary.create_file_writer(os.path.join(tensorboard_subdir, f'fit_{fit_nb}'))

            print(f'Fit {fit_nb+1}/{num_fits}')
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            key_montecarlo, subkey_montecarlo = jax.random.split(key_montecarlo, 2)

            params, avg_elbos = self.fit(subkey_params, subkey_batcher, subkey_montecarlo, data, log_writer, args)

            best_epoch = jnp.nanargmax(jnp.array(avg_elbos))
            best_epochs.append(best_epoch)
            best_elbo = avg_elbos[best_epoch]
            best_elbos.append(best_elbo)
            print(f'Best ELBO {best_elbo:.3f} at epoch {best_epoch}')
        
            if store_every != 0:
                selected_epochs = list(range(0, self.num_epochs, store_every))
                all_params.append({epoch_nb:params[epoch_nb] for epoch_nb in selected_epochs})

            else: 
                all_params.append(params[best_epoch])
            all_avg_elbos.append(avg_elbos)


        best_optim = jnp.argmax(jnp.array(best_elbos))
        print(f'Best fit is {best_optim+1}.')
        best_params = all_params[best_optim]

        if store_every != 0: 
            return all_params[best_optim], all_avg_elbos[best_optim]
        else: 
            return best_params, (best_optim, best_epochs, all_avg_elbos)

