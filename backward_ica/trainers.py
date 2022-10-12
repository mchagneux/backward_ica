from backward_ica.hmm import * 
from backward_ica.elbos import * 
import tensorflow as tf 
from jax.tree_util import tree_flatten

def winsorize_grads():
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        flattened_updates = jnp.concatenate([arr.flatten() for arr in tree_flatten(updates)[0]])
        high_value = jnp.sort(jnp.abs(flattened_updates))[int(0.90*flattened_updates.shape[0])]
        return jax.tree_map(lambda x: jnp.clip(x, -high_value, high_value), updates), ()
    return optax.GradientTransformation(init_fn, update_fn)


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
                sweep_sequence=False):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.q = q 
        # self.q.print_num_params()
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
        self.sweep_sequence = sweep_sequence
        # format_params = lambda params: self.q.format_params(params)


        if online:
            self.elbo = OnlineGeneralBackwardELBO(self.p, self.q, exp_and_normalize, num_samples)
            self.get_montecarlo_keys = get_keys
            self.loss = lambda key, data, params: -self.elbo(key, data, self.p.format_params(self.theta_star), q.format_params(params))[0]
        else:
            if force_full_mc: 
                self.elbo = GeneralBackwardELBO(self.p, self.q, num_samples)
                self.get_montecarlo_keys = get_keys
                self.loss = lambda key, data, params: -self.elbo(key, data, self.p.format_params(self.theta_star), q.format_params(params))
            else:
                if isinstance(self.p, LinearGaussianHMM):
                    self.elbo = LinearGaussianELBO(self.p, self.q)
                    self.get_montecarlo_keys = get_dummy_keys
                    self.loss = lambda key, data, params: -self.elbo(data, self.p.format_params(self.theta_star), q.format_params(params))
                elif isinstance(self.q, LinearBackwardSmoother) and self.p.transition_kernel.map_type == 'linear':
                    self.elbo = BackwardLinearELBO(self.p, self.q, num_samples)
                    self.get_montecarlo_keys = get_keys
                    self.loss = lambda key, data, params: -self.elbo(key, data, self.p.format_params(self.theta_star), q.format_params(params))
                else:
                    self.elbo = GeneralBackwardELBO(self.p, self.q, num_samples)
                    self.get_montecarlo_keys = get_keys
                    self.loss = lambda key, data, params: -self.elbo(key, data, self.p.format_params(self.theta_star), q.format_params(params))

    def fit(self, key_params, key_batcher, key_montecarlo, data, log_writer=None, args=None):


        num_seqs = data.shape[0]
        seq_length = len(data[0])

        # theta = self.p.get_random_params(key_theta, args)
        params = self.q.get_random_params(key_params, args)

        params = tree_map(lambda param, frozen_param: param if frozen_param == '' else frozen_param, 
                        params, 
                        self.frozen_params)

        opt_state = self.optimizer.init(params)
        subkeys = self.get_montecarlo_keys(key_montecarlo, num_seqs, self.num_epochs)

        if self.sweep_sequence: 
            # timesteps = jnp.arange(0, seq_length)
            def step(carry, x):

                def batch_step(params, opt_state, batch, keys):
                    avg_elbo_batch_timesteps = jnp.empty((seq_length-1,))
                    for i, timestep in enumerate(range(2,seq_length+1)):
                        batch_up_to_timestep = jax.lax.dynamic_slice_in_dim(batch, 0, timestep, axis=1)
                        neg_elbo_values, grads = jax.vmap(jax.value_and_grad(self.loss, argnums=2), in_axes=(0,0,None))(keys, batch_up_to_timestep, params)
                        avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), grads)
                        updates, opt_state = self.optimizer.update(avg_grads, opt_state, params)
                        params = optax.apply_updates(params, updates)
                        avg_elbo_batch_timesteps = avg_elbo_batch_timesteps.at[i].set(-jnp.mean(neg_elbo_values / batch_up_to_timestep.shape[1]))

                    return params, opt_state, jnp.mean(avg_elbo_batch_timesteps)

                data, params, opt_state, subkeys_epoch = carry
                batch_start = x
                batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
                batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)

                params, opt_state, avg_elbo_batch = batch_step(params, opt_state, batch_obs_seq, batch_keys)

                return (data, params, opt_state, subkeys_epoch), avg_elbo_batch
        else: 
            def step(carry, x):
                def batch_step(params, opt_state, batch, keys):

                    neg_elbo_values, grads = jax.vmap(jax.value_and_grad(self.loss, argnums=2), in_axes=(0,0,None))(keys, batch, params)
                    avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), grads)
                    
                    updates, opt_state = self.optimizer.update(avg_grads, opt_state, params)
                    params = optax.apply_updates(params, updates)
                    return params, \
                        opt_state, \
                        -jnp.mean(neg_elbo_values / seq_length)

                data, params, opt_state, subkeys_epoch = carry
                batch_start = x
                batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
                batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)
                params, opt_state, avg_elbo_batch = batch_step(params, opt_state, batch_obs_seq, batch_keys)
                return (data, params, opt_state, subkeys_epoch), avg_elbo_batch


        avg_elbos = []
        all_params = []
        batch_start_indices = jnp.arange(0, num_seqs, self.batch_size)

        t = tqdm(total=self.num_epochs, desc='Epoch')
        for epoch_nb in range(self.num_epochs):
            t.update(1)
            subkeys_epoch = subkeys[epoch_nb]
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            
            data = jax.random.permutation(subkey_batcher, data)
        

            (_ , params, opt_state, _), avg_elbo_batches = jax.lax.scan(f=step,  
                                                                        init=(data, params, opt_state, subkeys_epoch), 
                                                                        xs = batch_start_indices)


            avg_elbo_epoch = jnp.mean(avg_elbo_batches)
            t.set_postfix({'Avg ELBO epoch':avg_elbo_epoch})

            # avg_grads_batches = [grad for mask, grad in zip(tree_flatten(self.trainable_params)[0], 
            #                                                 tree_flatten(avg_grads_batches)[0]) 
            #                                             if mask]



            
            if log_writer is not None:
                with log_writer.as_default():
                    tf.summary.scalar('Epoch ELBO', avg_elbo_epoch, epoch_nb)
                    for batch_nb, avg_elbo_batch in enumerate(avg_elbo_batches):
                        tf.summary.scalar('Minibatch ELBO', avg_elbo_batch, epoch_nb*len(batch_start_indices) + batch_nb)
                        # avg_grads_batch = jnp.concatenate([grad[batch_nb].flatten() for grad in avg_grads_batches])
                        # sns.violinplot(avg_grads_batch)
                        # sns.swarmplot(avg_grads_batch)
                        # plt.savefig(os.path.join('grads',f'{epoch_nb*len(batch_start_indices) + batch_nb}'))
                        # tf.summary.image('Minibatch grads histogram', plot_to_image(plt.gcf()), epoch_nb*len(batch_start_indices) + batch_nb)
                        # plt.clf()
                        # tf.summary.histogram('Minibatch grads', avg_grads_batch, epoch_nb*len(batch_start_indices) + batch_nb)
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
        
            if store_every is not None:
                selected_epochs = list(range(0, self.num_epochs, store_every))
                all_params.append({epoch_nb:params[epoch_nb] for epoch_nb in selected_epochs})

            else: 
                all_params.append(params[best_epoch])
            all_avg_elbos.append(avg_elbos)


        best_optim = jnp.argmax(jnp.array(best_elbos))
        print(f'Best fit is {best_optim+1}.')
        best_params = all_params[best_optim]

        if store_every is not None: 
            return best_params, all_avg_elbos[best_optim]
        else: 
            return best_params, (best_optim, best_epochs, all_avg_elbos)

    def profile(self, key, data, theta):

        params_key, monte_carlo_key = jax.random.split(key, 2)
        phi = self.q.get_random_params(params_key)

        if self.use_johnson: 
            aux_params = self.aux_init_params(params_key, data[0][0])
        else:
            aux_params = None        
        if self.use_johnson: 
            loss = lambda seq, key, phi, aux_params: self.loss(seq, key, self.p.format_params(theta), self.q.format_params(phi), aux_params)
            params = (phi, aux_params)
        else: 
            loss = lambda seq, key, phi: self.loss(seq, key, self.p.format_params(theta), self.q.format_params(phi), None)
            params = phi

        num_seqs = data.shape[0]
        subkeys = self.get_montecarlo_keys(monte_carlo_key, num_seqs, 1).squeeze()

        @jit
        def step(params, batch, keys):
            return jax.vmap(loss, in_axes=(0,0,None))(batch, keys, params)

        step(params, data[:self.batch_size], subkeys[:self.batch_size])

        with jax.profiler.trace('./profiling/'):
            print(step(params, data[:2], subkeys[:2]))

    def check_elbo(self, data, theta):
        if isinstance(self.p, LinearGaussianHMM):
            print('Checking ELBO quality...')

            avg_evidences = vmap(jit(lambda seq: self.p.likelihood_seq(seq, theta)))(data)
            theta = self.p.format_params(theta)
            if isinstance(self.elbo, LinearGaussianELBO):
                elbo = jit(lambda key, seq:self.elbo(seq, theta, theta))
            else: 
                elbo = jit(lambda key, seq: self.elbo(key, seq, theta, theta))
            keys = jax.random.split(jax.random.PRNGKey(0), data.shape[0])
            avg_elbos = vmap(elbo)(keys, data)
            print('Avg error with Kalman evidence:', jnp.mean(jnp.abs(avg_evidences-avg_elbos)))