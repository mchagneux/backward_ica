from backward_ica.offline_smoothing import *
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
                online=False,
                online_elbo=False):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.q = q 
        self.q.print_num_params()
        self.p = p 
        
        self.formatted_theta_star = self.p.format_params(theta_star)
        self.frozen_params = frozen_params

        if online: 
            learning_rate = learning_rate / seq_length

        self.trainable_params = tree_map(lambda x: x == '', self.frozen_params)
        self.fixed_params = tree_map(lambda x: x != '', self.frozen_params)


        base_optimizer = optax.apply_if_finite(getattr(optax, optimizer)(learning_rate),
                                            max_consecutive_errors=10)

        # base_optimizer = optax.apply_if_finite(
        #                             optax.masked(getattr(optax, optimizer)(learning_rate), 
        #                                                     self.trainable_params), 
        #                                     max_consecutive_errors=10)
        
        # zero_grads_optimizer = optax.masked(optax.set_to_zero(), self.fixed_params)

        self.optimizer = optax.chain(
                                optax.clip(1.0), 
                                base_optimizer)
        self.online = online

        self.online_elbo = online_elbo

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
            if online_elbo: 
                self.elbo = OnlinePaRISELBO(self.p, self.q, num_samples)
                self.offline_elbo_for_check = GeneralBackwardELBO(self.p, self.q, num_samples)
                self.get_montecarlo_keys = get_keys
                def online_elbo(key, data, compute_up_to, params):

                    offline_value, _  = self.offline_elbo_for_check(
                                                                key, 
                                                                data, 
                                                                compute_up_to,
                                                                self.formatted_theta_star, 
                                                                self.q.format_params(params))
                    carry, (ess, normalizing_csts, means_q_t, covs_q_t, eta1, eta2, particles) = self.elbo.batch_compute(key, 
                                                data, 
                                                self.formatted_theta_star, 
                                                params)
                    
                    return -(carry, (-offline_value, ess, normalizing_csts, means_q_t, covs_q_t, eta1, eta2, particles)
                self.loss = online_elbo
            else: 

                self.elbo = GeneralBackwardELBO(self.p, self.q, num_samples)
                self.get_montecarlo_keys = get_keys
                def offline_elbo(key, data, compute_up_to, params):
                    value, (means_q_t_offline, covs_q_t_offline, eta1_offline, eta2_offline) = self.elbo(
                                    key, 
                                    data, 
                                    compute_up_to, 
                                    self.formatted_theta_star, 
                                    q.format_params(params))
                    return -value, (means_q_t_offline, covs_q_t_offline, eta1_offline, eta2_offline)

                self.loss = offline_elbo
        if self.online: 

            self.elbo = OnlineELBOAndGradsReparam(p=self.p, 
                                    q=self.q, 
                                    num_samples=num_samples)
                                    
            self.get_montecarlo_keys = get_keys

            def online_update(carry, key, data, timestep, params):
                jac_Omega_tm1 = carry['stats']['jac_Omega']


                carry['theta'] = self.formatted_theta_star

                input = {'key': key, 
                        't':timestep, 
                        'phi':params,
                        'y':data}

                new_carry = self.elbo.compute(carry, input)[0]

                Omega_t, jac_Omega_t = new_carry['stats']['Omega'], new_carry['stats']['jac_Omega']

                neg_elbo_t = tree_map(lambda x: -jnp.mean(x, axis=0) / (timestep + 1), Omega_t)

                grad_neg_elbo_t = tree_map(lambda x,y: jnp.mean(x-y, axis=0) / (timestep+1), jac_Omega_tm1, jac_Omega_t)
                
                return new_carry, (neg_elbo_t, grad_neg_elbo_t)



            self.online_update = online_update
            self.online_init = partial(self.elbo.init_carry, params=self.q.get_random_params(jax.random.PRNGKey(0)))

        if self.online:

            init_elbo_carry = vmap(self.online_init, 
                                    axis_size=self.batch_size)()
        

            def step(carry, x):
                def batch_step(params, opt_state, batch, keys):
                    def timestep_step(carry, x):
                        elbo_carry, params, opt_state = carry 
                        t, keys_t, obs_batch_t = x
                        elbo_carry, (neg_elbo_values, grads) = vmap(self.online_update, in_axes=(0,0,0,None,None))(
                                                                elbo_carry, 
                                                                keys_t, 
                                                                obs_batch_t, 
                                                                t,
                                                                params)

                        avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), grads)
                        updates, opt_state = self.optimizer.update(avg_grads, opt_state, params)
                        params = optax.apply_updates(params, updates)
                        return (elbo_carry, params, opt_state), neg_elbo_values


                    keys = vmap(partial(jax.random.split, num=batch.shape[1]))(keys)

                    xs = (self.timesteps(batch.shape[1]), 
                            jnp.transpose(keys, (1,0,2)), 
                            jnp.transpose(batch, (1,0,2)))


                    (elbo_carry, params, opt_state), neg_elbo_values = jax.lax.scan(
                                                                f=timestep_step, 
                                                                init=(init_elbo_carry, 
                                                                    params, 
                                                                    opt_state), 
                                                                xs=xs)
                    
                    return params, opt_state, jnp.mean(neg_elbo_values, axis=0)

                data, params, opt_state, subkeys_epoch = carry
                batch_start = x
                batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
                batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)

                params, opt_state, avg_elbo_batch = batch_step(
                                                            params, 
                                                            opt_state, 
                                                            batch_obs_seq, 
                                                            batch_keys)

                return (data, params, opt_state, subkeys_epoch), avg_elbo_batch


        else: 
            def step(carry, x):
                def batch_step(params, opt_state, batch, keys):
                    (neg_elbo_values, aux), grads = vmap(value_and_grad(self.loss, argnums=3, has_aux=True), in_axes=(0, 0, None, None))(keys, batch, batch.shape[1]-1, params)

                    avg_grads = jax.tree_util.tree_map(partial(jnp.mean, axis=0), grads)
                    updates, opt_state = self.optimizer.update(avg_grads, opt_state, params)
                    params = optax.apply_updates(params, updates)

                    
                    aux = [jnp.mean(aux_elem, axis=0) for aux_elem in aux]

                    if self.online_elbo:
                        aux[0] = -aux[0]



                    return params, \
                        opt_state, \
                        -jnp.mean(neg_elbo_values), ravel_pytree(avg_grads)[0], aux

                data, params, opt_state, subkeys_epoch = carry

                batch_start = x
                batch_obs_seq = jax.lax.dynamic_slice_in_dim(data, batch_start, self.batch_size)
                batch_keys = jax.lax.dynamic_slice_in_dim(subkeys_epoch, batch_start, self.batch_size)

                params, opt_state, avg_elbo_batch, avg_grads_batch, aux = batch_step(params, opt_state, batch_obs_seq, batch_keys)

                return (data, params, opt_state, subkeys_epoch), (avg_elbo_batch, avg_grads_batch, aux)
            

        self.step = step

    def timesteps(self, seq_length):
        return jnp.arange(0, seq_length)

    def fit(self, key_params, key_batcher, key_montecarlo, data, log_writer=None, args=None, log_writer_monitor=None):


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
        aux_list = []
        for epoch_nb in range(self.num_epochs):
            t.update(1)
            subkeys_epoch = subkeys[epoch_nb]
            key_batcher, subkey_batcher = jax.random.split(key_batcher, 2)
            
            data = jax.random.permutation(subkey_batcher, data)
        

            (_ , params, opt_state, _), (avg_elbo_batches, avg_grads_batches, aux) = jax.lax.scan(f=self.step,  
                                                                        init=(
                                                                            data, 
                                                                            params, 
                                                                            opt_state, 
                                                                            subkeys_epoch), 
                                                                        xs=batch_start_indices)


            # print(jnp.linalg.norm(jax.flatten_util.ravel_pytree(avg_grads)[0]))
            avg_elbo_epoch = jnp.mean(avg_elbo_batches)
            avg_grads_epoch = jnp.mean(avg_grads_batches, axis=0)
            if self.online_elbo:
                avg_monitor_elbo_batches = aux[0]
                avg_monitor_elbo_epoch = jnp.mean(avg_monitor_elbo_batches)
                avg_ess_epoch = jnp.mean(aux[1], axis=0)[1:]
                avg_normalizing_cst_epoch = jnp.mean(aux[2], axis=0)[1:]


                avg_means_epoch = jnp.mean(aux[3], axis=0)
                avg_covs_epoch = jnp.mean(aux[4], axis=0)
                avg_eta1_epoch = jnp.mean(aux[5], axis=0)[1:]
                avg_eta2_epoch = jnp.mean(aux[6], axis=0)[1:]
                particles = jnp.mean(aux[7], axis=0)


            else:
                avg_means_epoch = jnp.mean(aux[0], axis=0)
                avg_covs_epoch = jnp.mean(aux[1], axis=0)
                avg_eta1_epoch = jnp.mean(aux[2], axis=0)[:-1]
                avg_eta2_epoch = jnp.mean(aux[3], axis=0)[:-1]
                
            
            aux_list.append([params, avg_means_epoch, avg_covs_epoch, avg_eta1_epoch, avg_eta2_epoch])


            # norm_type = 'min'

            # if norm_type == 'min':
            #     norm = lambda x: jnp.min(x, axis=-1)
            # else: 
            #     norm = lambda x: jnp.linalg.norm(x, axis=-1, ord=norm_type)

            # norm_means = norm(avg_means_epoch)
            # max_norm_means = norm(norm_means)

            # norm_covs = norm(avg_covs_epoch)
            # max_norm_covs = norm(norm_covs)

            # norm_eta_1 = norm(norm(avg_eta1_epoch))
            # max_eta_1 = norm(norm_eta_1)

            # norm_eta_2 = norm(norm(-avg_eta2_epoch))
            # max_eta_2 = norm(norm_eta_2)

            t.set_postfix({'Avg ELBO epoch':avg_elbo_epoch})

            if log_writer is not None:
                with log_writer.as_default():
                    tf.summary.scalar('Epoch ELBO', avg_elbo_epoch, epoch_nb)
                    # tf.summary.histogram('Histogram gradients', avg_grads_epoch, epoch_nb)
                    if self.online_elbo:
                        tf.summary.scalar('Max ESS (over t and particles)', jnp.max(avg_ess_epoch), epoch_nb)
                        tf.summary.scalar('Min ESS (over t and particles)', jnp.min(avg_ess_epoch), epoch_nb)
                        tf.summary.scalar('Mean ESS (over t and particles)', jnp.mean(avg_ess_epoch), epoch_nb)
                        tf.summary.scalar('Min position particles', jnp.min(particles), epoch_nb)
                        tf.summary.scalar('Max position particles', jnp.max(particles), epoch_nb)

                        # tf.summary.scalar('Mean (over t and particles) normalizing cst', jnp.mean(mean_normalizing_const, axis=0), epoch_nb)
                        # tf.summary.scalar('Mean (over t) of min (over particles) normalizing cst', jnp.mean(min_normalizing_const, axis=0), epoch_nb)
                        tf.summary.scalar('Mean (over t and particles) normalizing cst', jnp.mean(avg_normalizing_cst_epoch), epoch_nb)
                        tf.summary.scalar('Min (over t and particles) normalizing cst', jnp.min(avg_normalizing_cst_epoch), epoch_nb)
                        tf.summary.scalar('Max (over t and particles) normalizing cst', jnp.max(avg_normalizing_cst_epoch), epoch_nb)


                    tf.summary.histogram('mu', avg_means_epoch, epoch_nb)
                    tf.summary.histogram('diag(Sigma)', avg_covs_epoch, epoch_nb)

                    tf.summary.histogram('eta_1', avg_eta1_epoch, epoch_nb)
                    tf.summary.histogram('diag(-eta_2)', -avg_eta2_epoch, epoch_nb)

                    for batch_nb, avg_elbo_batch in enumerate(avg_elbo_batches):
                        tf.summary.scalar('Minibatch ELBO', 
                                          avg_elbo_batch, 
                                          epoch_nb*len(batch_start_indices) + batch_nb)
                        
            if log_writer_monitor is not None:
                with log_writer_monitor.as_default():
                    tf.summary.scalar('Epoch ELBO', avg_monitor_elbo_epoch, epoch_nb)
                    # tf.summary.histogram('ESS', avg_ess_epoch, epoch_nb)
                    # tf.summary.scalar('Min ESS', jnp.min(avg_ess_epoch), epoch_nb)
    
                    # print(avg_ess_epoch.shape)
                    for batch_nb, avg_monitor_elbo_batch in enumerate(avg_monitor_elbo_batches):
                        tf.summary.scalar( 
                                        'Minibatch ELBO', 
                                        avg_monitor_elbo_batch, 
                                        epoch_nb*len(batch_start_indices) + batch_nb)
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
            if self.online_elbo:
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
                all_params.append(params[best_epoch])
            all_avg_elbos.append(avg_elbos)


        best_optim = jnp.argmax(jnp.array(best_elbos))
        print(f'Best fit is {best_optim+1}.')
        best_params = all_params[best_optim]

        if store_every != 0: 
            return all_params[best_optim], all_avg_elbos[best_optim]
        else: 
            return best_params, (best_optim, best_epochs, all_avg_elbos)

