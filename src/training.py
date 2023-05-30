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

    def __init__(self, 
                p:HMM, 
                theta_star,
                q:BackwardSmoother, 
                optimizer, 
                learning_rate, 
                optim_options,
                num_epochs, 
                batch_size, 
                seq_length,
                num_samples=1, 
                force_full_mc=False,
                frozen_params=None,
                training_mode='offline',
                elbo_mode='autodiff_on_backward'):
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.q = q 
        self.q.print_num_params()
        self.p = p 
        
        self.formatted_theta_star = self.p.format_params(theta_star)
        self.frozen_params = frozen_params

        self.elbo_mode = elbo_mode

        self.training_mode = training_mode
        if 'true_online' in training_mode:
            true_online = True
            self.online_batch_size = 1
            self.online_reset = False
                

        else: 
            true_online = False
            self.online_batch_size = int(training_mode.split(',')[1])
            self.online_reset = 'reset' in training_mode
        
        if elbo_mode != 'autodiff_on_backward':
            learning_rate *= self.online_batch_size / 100

        self.true_online = true_online


        self.monitor = False

        self.elbo_options = {}
        if 'score' in elbo_mode: 
            for option in ['paris', 'variance_reduction', 'mcmc']:
                self.elbo_options[option] = True if option in elbo_mode else False

            if 'bptt_depth' in elbo_mode: 
                self.elbo_options['bptt_depth'] = int(elbo_mode.split('bptt_depth')[1].split('_')[1])

            self.elbo_options['true_online'] = True if self.true_online else False
        
        def optimizer_update_fn(params, updates):
            new_params = optax.apply_updates(params, updates)
            # if self.online:
            #     new_params = optax.incremental_update(new_params, params, step_size=0.8)
            return new_params
        
        self.optimizer_update_fn = optimizer_update_fn


        self.trainable_params = tree_map(lambda x: x == '', self.frozen_params)
        self.fixed_params = tree_map(lambda x: x != '', self.frozen_params)



        if 'linear_sched' in optim_options:
            
            learning_rate = optax.linear_schedule(learning_rate, 
                                                  end_value=learning_rate / 100, 
                                                  transition_steps=num_epochs * (seq_length / self.online_batch_size))
        
        elif 'warmup_cosine' in optim_options:

            learning_rate = optax.warmup_cosine_decay_schedule(
                init_value=learning_rate / 100,
                peak_value=learning_rate / 10,
                warmup_steps=20 * seq_length / self.online_batch_size,
                decay_steps=num_epochs * (seq_length / self.online_batch_size),
                end_value=learning_rate / 1000,
                )
        elif 'cst' in optim_options:
            pass 

        
        else:
            raise NotImplementedError
                
        base_optimizer = optax.apply_if_finite(getattr(optax, optimizer)(learning_rate),
                                            max_consecutive_errors=10)


        self.optimizer = base_optimizer

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
                
                elbo, grad = self.elbo.postprocess(carry) 
                T = len(ys) - 1 
                neg_elbo = -elbo / (T+1)
                neg_grad = tree_map(lambda x: -x / (T+1), grad)
                return (neg_elbo, neg_grad), aux
                    
            self.elbo_step = self.elbo.step
            
                
        elif ('autodiff_on_backward' in self.elbo_mode) and self.online_reset:
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
                return (neg_elbo, neg_grad), aux

        else:
            print('ELBO mode not suitable for gradient accumulation.')
            raise NotImplementedError

        self.elbo_batch = elbo_and_grads_batch
        self.get_montecarlo_keys = get_keys

        if self.online_reset:
            init_carry = 0.0
        else: 
            init_carry = self.elbo.init_carry(self.q.get_random_params(jax.random.PRNGKey(0)))

        self.init_carry = init_carry

        def online_update(key, 
                        elbo_carry, 
                        strided_ys, 
                        timesteps, 
                        params):
            
            if self.online_reset:
                
                (neg_elbo, neg_grad), aux = self.elbo_batch(key, strided_ys, params)

            else: 
                keys = jax.random.split(key, len(timesteps))

                if self.true_online: 
                    keys = jax.random.split(key, len(timesteps))

                    def _step(carry, x):
                        key, t, strided_y = x
                        input_t = {'t':t, 
                                'key': key, 
                                'ys_bptt':strided_y, 
                                'T':timesteps[-1],
                                'phi':params}
                        
                        carry['theta'] = self.formatted_theta_star
                        new_carry, aux = self.elbo.step(carry, 
                                                    input_t)
                    
                        elbo_t, grad_t = self.elbo.postprocess(new_carry)

                        return new_carry, (aux, new_carry, elbo_t, grad_t)
                    
                    _ , (aux, carries, elbos, grads) = jax.lax.scan(_step, 
                                                                    init=elbo_carry,
                                                                    xs=(keys, 
                                                                        timesteps, 
                                                                        strided_ys))
                    elbo_carry = tree_get_idx(0, carries)

                    neg_elbo = -elbos[-1] / (timesteps[-1]+1)
                    grad_tm1 = tree_get_idx(0, grads)
                    grad_t = tree_get_idx(1, grads)
                    neg_grad = tree_map(lambda x,y: -(x-y), grad_t, grad_tm1)
                        

                else: 

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
                    
                
                    elbo, grad = self.elbo.postprocess(elbo_carry)
                    T = timesteps[-1]
                    neg_elbo = -elbo / (T+1)
                    neg_grad = tree_map(lambda x: -x / (T+1), grad)

        
            return neg_elbo, neg_grad, elbo_carry, aux
                    

        @jax.jit
        def step(key, strided_data_on_timesteps, elbo_carry, timesteps, params, opt_state):

            neg_elbo, grad, elbo_carry, aux = online_update(
                                                            key, 
                                                            elbo_carry, 
                                                            strided_data_on_timesteps, 
                                                            timesteps, 
                                                            params)

            
            updates, opt_state = self.optimizer.update(grad, 
                                                        opt_state, 
                                                        params)
            
            params = self.optimizer_update_fn(params, updates)

            return (params, opt_state, elbo_carry), (-neg_elbo, ravel_pytree(grad)[0], aux)
        
        self.step = step

    def timesteps(self, seq_length, delta):
        if self.true_online: 
            return tree_get_strides(2, jnp.arange(0, seq_length))

        else:
            return jnp.array(jnp.array_split(jnp.arange(0, seq_length), 
                                            seq_length // delta))

    def fit(self, key_params, key_batcher, key_montecarlo, data, log_writer=None, args=None, log_writer_monitor=None):



        params = self.q.get_random_params(key_params, args)

        params = tree_map(lambda param, frozen_param: param if frozen_param == '' else frozen_param, 
                        params, 
                        self.frozen_params)

        opt_state = self.optimizer.init(params)

        avg_elbos = []
        all_params = []
        aux_list = []

        t = tqdm(total=self.num_epochs, desc='Epoch')
        data = data[0]
        seq_length = data.shape[0]
        plot_step_size = seq_length
        keys = get_keys(key_montecarlo, seq_length // self.online_batch_size, self.num_epochs)
        with log_writer.as_default():
            for epoch_nb, keys_epoch in enumerate(keys):
                elbo_carry = self.init_carry
                strided_data = self.elbo.preprocess(data)
                for step_nb, (timesteps, key_step) in enumerate(zip(self.timesteps(data.shape[0], self.online_batch_size), keys_epoch)):
                    
                    (params, opt_state, elbo_carry), (elbo, _ , _) = self.step(key_step, 
                                                                                strided_data[timesteps], 
                                                                                elbo_carry, 
                                                                                timesteps, 
                                                                                params, 
                                                                                opt_state)
                    tf.summary.scalar('ELBO', 
                                    elbo, 
                                    epoch_nb*plot_step_size + step_nb * self.online_batch_size)
            
            all_params.append(params)
            avg_elbos.append(elbo)
            t.update(1)

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
            if self.monitor:
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

