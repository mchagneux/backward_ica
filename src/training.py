from src.offline_smoothing import *
from src.online_smoothing import *
from src.stats.hmm import * 
from src.variational.sequential_models import *

import tensorflow as tf 
from jax.tree_util import tree_flatten
import jax
from jax import vmap, value_and_grad, numpy as jnp

import optax 

# def winsorize_grads():
#     def init_fn(_): 
#         return ()
#     def optimizer_update_fn(updates, state, params=None):
#         flattened_updates = jnp.concatenate([arr.flatten() for arr in tree_flatten(updates)[0]])
#         high_value = jnp.sort(jnp.abs(flattened_updates))[int(0.90*flattened_updates.shape[0])]
#         return tree_map(lambda x: jnp.clip(x, -high_value, high_value), updates), ()
#     return optax.GradientTransformation(init_fn, update_fn)

def define_frozen_tree(key, frozen_params, q, theta_star):

    # key_theta, key_phi = random.split(key, 2)

    frozen_phi = q.get_random_params(key)
    frozen_phi = tree_map(lambda x: '', frozen_phi)


    if 'prior' in frozen_params:
        if isinstance(q, LinearGaussianHMM) or isinstance(q, JohnsonSmoother):
            frozen_phi.prior = theta_star.prior
        elif isinstance(q, NeuralBackwardSmoother):
            frozen_phi.prior = q.frozen_prior()

    if 'covariances' in frozen_params: 
        frozen_phi.transition.noise.scale = theta_star.transition.noise.scale
    
    return frozen_phi
    
class SVITrainer:

    def __init__(self, p:HMM, 
                theta_star,
                q:BackwardSmoother, 
                optimizer, 
                learning_rate, 
                num_epochs, 
                batch_size, 
                seq_length,
                num_samples=1, 
                force_full_mc=False,
                frozen_params=None,
                elbo_mode='autodiff_on_backward',
                online=False,
                online_batch_size=10):
        
        self.online_batch_size = online_batch_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.q = q 
        self.q.print_num_params()
        self.p = p 
        
        self.formatted_theta_star = self.p.format_params(theta_star)
        self.frozen_params = frozen_params

        self.elbo_mode = elbo_mode
        self.online = online

        if online:
            if 'polyak' in self.elbo_mode:
                self.polyak_averaging = True
            else: 
                self.polyak_averaging = False
            self.online_reset = 'reset' in elbo_mode
        
        self.monitor = False

        self.elbo_options = {}
        if 'score' in elbo_mode: 
            for option in ['paris', 'variance_reduction']:
                self.elbo_options[option] = True if option in elbo_mode else False

            if 'bptt_depth' in elbo_mode: 
                self.elbo_options['bptt_depth'] = int(elbo_mode.split('bptt_depth')[1].split('_')[1])

        def optimizer_update_fn(params, updates):
            new_params = optax.apply_updates(params, updates)
            # if self.online:
            #     new_params = optax.incremental_update(new_params, params, step_size=0.8)
            return new_params
        
        self.optimizer_update_fn = optimizer_update_fn


        self.trainable_params = tree_map(lambda x: x == '', self.frozen_params)
        self.fixed_params = tree_map(lambda x: x != '', self.frozen_params)

        learning_rate = optax.linear_schedule(learning_rate, 
                                              end_value=learning_rate / 100, 
                                              transition_steps=num_epochs * self.online_batch_size)
        
        base_optimizer = optax.apply_if_finite(getattr(optax, optimizer)(learning_rate),
                                            max_consecutive_errors=10)

        # base_optimizer = optax.apply_if_finite(
        #                             optax.masked(getattr(optax, optimizer)(learning_rate), 
        #                                                     self.trainable_params), 
        #                                     max_consecutive_errors=10)
        
        # zero_grads_optimizer = optax.masked(optax.set_to_zero(), self.fixed_params)

        self.optimizer = optax.chain(
                                # optax.clip(1.0),
                                # zero_grads_optimizer, 
                                base_optimizer)


        if not self.online: 

            if isinstance(self.p, LinearGaussianHMM) and isinstance(self.q, LinearGaussianHMM) and (not force_full_mc):
                self.elbo = LinearGaussianELBO(self.p, self.q)
                self.get_montecarlo_keys = get_dummy_keys
                def closed_form_elbo(key, data, compute_up_to, params):
                    value, dummy_aux = self.elbo(
                                        data, 
                                        compute_up_to, 
                                        self.formatted_theta_star, 
                                        q.format_params(params))
                    return -value, dummy_aux
                self.loss = closed_form_elbo
            else: 
                if 'score' in self.elbo_mode:
                    self.elbo = OnlineELBOScoreGradients(
                                                        self.p, 
                                                        self.q, 
                                                        num_samples=num_samples, 
                                                        **self.elbo_options)

                    self.get_montecarlo_keys = get_keys
                    def online_elbo(key, data, compute_up_to, params):
                        carry, aux = self.elbo.batch_compute(key, 
                                                    data, 
                                                    self.formatted_theta_star, 
                                                    params)
                        neg_elbo, neg_grad = self.elbo.postprocess(carry, 
                                                           T=compute_up_to)
                        return (neg_elbo, neg_grad), aux
                    self.loss = online_elbo

                elif 'autodiff_on_backward' in self.elbo_mode: 
                    self.elbo = GeneralBackwardELBO(self.p, self.q, num_samples)
                    self.get_montecarlo_keys = get_keys
                    def offline_elbo(key, data, compute_up_to, params):
                        value, aux = self.elbo(
                                        key, 
                                        data, 
                                        compute_up_to, 
                                        self.formatted_theta_star, 
                                        q.format_params(params))
                        return -value / (compute_up_to + 1), aux

                    self.loss = offline_elbo

                else: 
                    print('ELBO mode not suitable for offline learning.')

                def batch_step(carry, x):
                    def step(params, opt_state, batch, keys):
                        if 'score' in self.elbo_mode:
                            (neg_elbo_values, grads), aux = vmap(self.loss, in_axes=(0, 0, None, None))(keys, 
                                                                                                        batch, 
                                                                                                        batch.shape[1]-1, 
                                                                                                        params)
                        else:
                            (neg_elbo_values, aux), grads = vmap(value_and_grad(self.loss, 
                                                                                argnums=3, 
                                                                                has_aux=True), 
                                                                in_axes=(0, 0, None, None))(keys, 
                                                                                            batch, 
                                                                                            batch.shape[1]-1, 
                                                                                            params)
                        
                        avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), grads)
                        updates, opt_state = self.optimizer.update(avg_grads, opt_state, params)

                        params = self.optimizer_update_fn(params, updates)

                        return params, \
                            opt_state, \
                            -jnp.mean(neg_elbo_values), ravel_pytree(avg_grads)[0], aux

                    data, params, opt_state, subkeys_epoch = carry

                    batch_start = x
                    batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
                    batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)

                    params, opt_state, avg_elbo_batch, avg_grads_batch, aux = step(params, opt_state, batch_obs_seq, batch_keys)

                    return (data, params, opt_state, subkeys_epoch), (avg_elbo_batch, avg_grads_batch, aux)
        
        else: 
            if self.monitor:
                self.monitor_elbo = GeneralBackwardELBO(p, q, num_samples)

            if 'score' in self.elbo_mode:
                print('USING SCORE ELBO.')
                self.elbo = OnlineELBOScoreGradients(
                                                self.p, 
                                                self.q, 
                                                num_samples=num_samples, 
                                                **self.elbo_options)
                
                        
                def elbo_and_grads_batch(key, ys, params):
                    carry, aux = self.elbo.batch_compute(
                                                key, 
                                                ys, 
                                                self.formatted_theta_star, 
                                                params)
                    
                    neg_elbo, neg_grad = self.elbo.postprocess(carry, 
                                                               T=len(ys)-1)
                    return (neg_elbo, neg_grad), jnp.mean(aux[1:-1])
                     
                self.elbo_step = self.elbo.step

            elif 'autodiff_on_forward' in self.elbo_mode and self.online_reset:
                print('USING AUTODIFF ON FORWARD ELBO.')

                self.elbo = OnlineELBO(p, q, num_samples, **self.elbo_options)
                
                def elbo_and_grads_batch(key, ys, params):
                    def f(params):
                        carry, aux = self.elbo.batch_compute(
                                            key, 
                                            ys, 
                                            self.formatted_theta_star, 
                                            params)
                        neg_elbo = self.elbo.postprocess(carry, T=len(ys)-1)
                        return neg_elbo, aux
                    (neg_elbo, aux), neg_grad = jax.value_and_grad(f, has_aux=True)(params)
                    return (neg_elbo, neg_grad), aux
                
                    
            elif 'autodiff_on_backward' in self.elbo_mode and self.online_reset:
                self.elbo = GeneralBackwardELBO(self.p, self.q, num_samples)
                print('USING AUTODIFF ON BACKWARD ELBO.')

                def elbo_and_grads_batch(key, ys, params):
                    def f(params):
                        elbo, aux = self.elbo(key, 
                                              ys, 
                                              len(ys)-1, 
                                              self.formatted_theta_star, 
                                              q.format_params(params))
                        return -elbo / len(ys), aux 
                    (neg_elbo, aux), neg_grad = jax.value_and_grad(f, has_aux=True)(params)
                    return (neg_elbo, neg_grad), jnp.mean(aux[1:-1])

            else: 
                print('ELBO mode not suitable for online learning.')

            self.elbo_batch = elbo_and_grads_batch
            self.get_montecarlo_keys = get_keys

            if not 'reset' in self.elbo_mode: 
                init_carry = jax.vmap(self.elbo.init_carry, 
                                        axis_size=batch_size, 
                                        in_axes=(None,))(self.q.get_random_params(jax.random.PRNGKey(0)))   

            else: 
                init_carry = jnp.empty((self.batch_size,1))

            def online_step(key, 
                            elbo_carry, 
                            data, 
                            timesteps, 
                            params):
                key, subkey = jax.random.split(key, 2)
                keys = jax.random.split(subkey, len(timesteps))
                
                if self.online_reset:
                    ys = data[timesteps]
                    
                    (neg_elbo, neg_grad), aux = self.elbo_batch(key, ys, params)

                else: 
                    strided_ys = self.elbo.preprocess(data)[timesteps]

                    def _step(carry, x):
                        key, t, ys_bptt = x
                        input_t = {'t':t, 
                                'key': key, 
                                'ys_bptt':ys_bptt, 
                                'T':timesteps[-1],
                                'phi':params}
                        
                        carry['theta'] = self.formatted_theta_star
                        carry, aux = self.elbo.step(carry, input_t)
                       
                        
                        return carry, aux
                
                    elbo_carry, aux = jax.lax.scan(_step, 
                                            init=elbo_carry, 
                                            xs=(keys, 
                                                timesteps, 
                                                strided_ys))
                    
                
                    neg_elbo, neg_grad = self.elbo.postprocess(elbo_carry, 
                                                    T=timesteps[-1])
                
           
                    
                return neg_elbo, neg_grad, key, elbo_carry, aux
            
            self.loss = online_step
            
            timesteps = self.timesteps(seq_length, 
                                       delta=self.online_batch_size)

            def batch_step(carry, x):
                data, params, opt_state, subkeys_epoch = carry
                batch_start = x
                data = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
                keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)
                def step(carry, timesteps):
                    keys, params, opt_state, elbo_carry = carry

                    neg_elbo_values, grads, keys, elbo_carry, aux = vmap(self.loss, 
                                                                        in_axes=(0, 0, 0, None, None))(
                                                                                                keys, 
                                                                                                elbo_carry, 
                                                                                                data, 
                                                                                                timesteps, 
                                                                                                params)



                    avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), 
                                                       grads)
                    
                    updates, opt_state = self.optimizer.update(avg_grads, 
                                                               opt_state, 
                                                               params)
                    params = self.optimizer_update_fn(params, updates)

                    carry = keys, params, opt_state, elbo_carry
                    out = -jnp.mean(neg_elbo_values), ravel_pytree(avg_grads)[0], jnp.mean(aux, axis=0)
                    return carry, out
                
                (_, params, opt_state, _), (avg_elbo_batch, avg_grads_batch, avg_aux_batch) = jax.lax.scan(
                                                                                        step, 
                                                                                        init=(keys, 
                                                                                            params, 
                                                                                            opt_state, 
                                                                                            init_carry),
                                                                                        xs=timesteps)
                # if self.monitor: 
                #     offline_elbo = jnp.mean(jax.vmap(self.monitor_elbo)(
                #                         key, 
                #                         data, 
                #                         len(data)-1,
                #                         self.formatted_theta_star, 
                #                         q.format_params(params), axis=0)[0] / len(data)
                # else: 
                #     offline_elbo = jnp.zeros_like(timesteps)

                return (data, params, opt_state, subkeys_epoch), (avg_elbo_batch, avg_grads_batch, avg_aux_batch)
        self.batch_step = batch_step

    def timesteps(self, seq_length, delta):

        return jnp.array(jnp.array_split(jnp.arange(0, seq_length), 
                                         seq_length // delta))

    def fit(self, key_params, key_batcher, key_montecarlo, data, log_writer=None, args=None, log_writer_monitor=None):


        num_seqs = data.shape[0]

        params = self.q.get_random_params(key_params, args)

        # with open(os.path.join('experiments/p_chaotic_rnn/2023_03_29__19_51_59/neural_backward__offline' ,'data'), 'rb') as f: 
        #     all_params_through_epochs = dill.load(f)


        # params = all_params_through_epochs[-1]


        params = tree_map(lambda param, frozen_param: param if frozen_param == '' else frozen_param, 
                        params, 
                        self.frozen_params)

        opt_state = self.optimizer.init(params)
        subkeys = self.get_montecarlo_keys(key_montecarlo, num_seqs, self.num_epochs)

        avg_elbos = []
        all_params = []
        batch_start_indices = jnp.arange(0, num_seqs, self.batch_size)

        t = tqdm(total=self.num_epochs, desc='Epoch')
        aux_list = []
        for epoch_nb in range(self.num_epochs):
            t.update(1)
            subkeys_epoch = subkeys[epoch_nb]
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            
            data = jax.random.permutation(subkey_batcher, data)

            (_ , params, opt_state, _), (avg_elbo_batches, avg_grads_batches, avg_aux_batches) = jax.lax.scan(
                                                                    f=self.batch_step,  
                                                                    init=(
                                                                        data, 
                                                                        params, 
                                                                        opt_state, 
                                                                        subkeys_epoch), 
                                                                    xs=batch_start_indices)
            


            # print(jnp.linalg.norm(jax.flatten_util.ravel_pytree(avg_grads)[0]))
            avg_aux_epoch = jnp.mean(avg_aux_batches)
            avg_elbo_epoch = jnp.mean(avg_elbo_batches, axis=0)

            if self.online: 
                if self.monitor: 
                    _, monitor_elbo_batches = avg_aux_batches
                    monitor_elbo_epoch = jnp.mean(monitor_elbo_batches, axis=0)[-1]
                    monitor_elbo_batches = monitor_elbo_batches[-1]

                avg_elbo_epoch = avg_elbo_epoch[-1]
                avg_elbo_batches = avg_elbo_batches[0]
                avg_aux_batches = avg_aux_batches[0]

                
            
            aux_list.append(params)

            t.set_postfix({'Avg ELBO epoch':avg_elbo_epoch})

            if log_writer is not None:
                with log_writer.as_default():
                    tf.summary.scalar('Epoch ELBO', avg_elbo_epoch, epoch_nb)
                    tf.summary.scalar('Epoch mean KL', avg_aux_epoch, epoch_nb)

                    if self.batch_size > 1:
                         for batch_nb, avg_elbo_batch in enumerate(avg_elbo_batches):
                            tf.summary.scalar('Minibatch ELBO', 
                                            avg_elbo_batch, 
                                            epoch_nb*self.batch_size + batch_nb)
                            
                    if self.online: 
                        step_size = len(avg_elbo_batches)*self.online_batch_size
                        for batch_nb, avg_elbo_batch in enumerate(avg_elbo_batches):
                            tf.summary.scalar('Temporal step ELBO', 
                                            avg_elbo_batch, 
                                            epoch_nb*step_size + batch_nb * self.online_batch_size)
                            
                        for batch_nb, avg_aux_batch in enumerate(avg_aux_batches):
                                                    tf.summary.scalar('Temporal step KL', 
                                                                    avg_aux_batch, 
                                                                    epoch_nb*step_size + batch_nb * self.online_batch_size)
            if log_writer_monitor is not None:
                with log_writer_monitor.as_default():
                    tf.summary.scalar('Epoch ELBO', monitor_elbo_epoch, epoch_nb)

                    step_size = len(avg_elbo_batches)*self.online_batch_size
                    for batch_nb, monitor_elbo_batch in enumerate(monitor_elbo_batches):
                        tf.summary.scalar('Temporal step ELBO', 
                                        monitor_elbo_batch, 
                                        epoch_nb*step_size + batch_nb * self.online_batch_size)
                        
            avg_elbos.append(avg_elbo_epoch)
            all_params.append(params)
        t.close()
                    
        return all_params, avg_elbos, aux_list

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
            if self.online and self.monitor:
                log_writer_monitor = tf.summary.create_file_writer(os.path.join(tensorboard_subdir, f'fit_{fit_nb}_monitor'))
            else:
                log_writer_monitor = None
            print(f'Fit {fit_nb+1}/{num_fits}')
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            key_montecarlo, subkey_montecarlo = jax.random.split(key_montecarlo, 2)

            params, avg_elbos, aux_list = self.fit(subkey_params, subkey_batcher, subkey_montecarlo, data, log_writer, args, log_writer_monitor=log_writer_monitor)
            with open(os.path.join(log_dir, 'data'), 'wb') as f:
                dill.dump(aux_list, f)
            best_epoch = jnp.nanargmax(jnp.array(avg_elbos))
            best_epochs.append(best_epoch)
            best_elbo = avg_elbos[best_epoch]
            best_elbos.append(best_elbo)
            print(f'Best ELBO {best_elbo:.3f} at epoch {best_epoch}')

            if store_every != 0:
                selected_epochs = list(range(0, self.num_epochs, store_every))
                all_params.append({epoch_nb:params[epoch_nb] for epoch_nb in selected_epochs})

            else: 
                all_params.append(params[-1])
            all_avg_elbos.append(avg_elbos)


        best_optim = jnp.argmax(jnp.array(best_elbos))
        print(f'Best fit is {best_optim+1}.')
        best_params = all_params[best_optim]

        if store_every != 0: 
            return all_params[best_optim], all_avg_elbos[best_optim]
        else: 
            return best_params, (best_optim, best_epochs, all_avg_elbos)

