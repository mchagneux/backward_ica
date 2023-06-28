from src.offline_smoothing import *
from src.online_smoothing import *
from src.stats.hmm import * 
from src.variational.sequential_models import *

import tensorflow as tf 
from jax.tree_util import tree_flatten
import jax
from jax import vmap, value_and_grad, numpy as jnp
import multiprocessing
import optax 
from time import time

def define_frozen_tree(key, frozen_params, q, theta_star):

    # key_theta, key_phi = random.split(key, 2)

    frozen_phi = q.get_random_params(key)
    frozen_phi = tree_map(lambda x: '', frozen_phi)

    if 'prior' in frozen_params:
        if isinstance(q, LinearGaussianHMM) or isinstance(q, JohnsonSmoother):
            frozen_phi.prior = theta_star.prior
        elif isinstance(q, NeuralBackwardSmoother):
            raise NotImplementedError


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
                frozen_params='',
                training_mode='offline',
                elbo_mode='autodiff_on_backward'):
        
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.q = q 
        self.q.print_num_params()
        self.p = p
        self.theta_star = theta_star
        self.formatted_theta_star = self.p.format_params(theta_star)
        self.frozen_params = frozen_params

        self.elbo_mode = elbo_mode
        self.training_mode = training_mode

        if 'share_params' in elbo_mode:
            print('Using transition and prior from true model.')
            if isinstance(q, LinearGaussianHMM):
                def build_params(params):
                    return HMM.Params(
                                    prior=theta_star.prior,
                                    transition=theta_star.transition,
                                    emission=params) 
                def extract_params(params):
                    return params.emission
            elif isinstance(q, JohnsonSmoother):
                def build_params(params):
                    return JohnsonParams(prior=theta_star.prior,
                                         transition=theta_star.transition,
                                         net=params)
                def extract_params(params):
                    return params.net
            elif isinstance(q, NeuralBackwardSmoother):
                def build_params(params):
                    return NeuralBackwardSmoother.Params(prior=params[0], 
                                                         backwd=theta_star.transition,
                                                         state=params[1],
                                                         filt=params[2])
                def extract_params(params):
                    return params.prior, params.state, params.filt
        else:
            def build_params(params):
                return params 
            def extract_params(params):
                return params
            
        self._build_params = build_params
        self._extract_params = extract_params

        if isinstance(p, LinearGaussianHMM) and isinstance(q, LinearGaussianHMM):
            print('Monitor ELBO is analytical.')
            monitor_elbo = LinearGaussianELBO(p, q)
            self.monitor_elbo = lambda _ , obs_seq, compute_up_to, theta, phi: monitor_elbo(obs_seq, 
                                                                                            compute_up_to,
                                                                                            theta, 
                                                                                            phi)
        
        else: 
            self.monitor_elbo = GeneralBackwardELBO(p, q, num_samples)

        if 'true_online' in training_mode:
            self.online_difference = 'difference' in training_mode
            self.num_grad_steps = int(training_mode.split(',')[1])
            true_online = True
            self.online_batch_size = 1
            self.reset = False
                
        else: 
            true_online = False
            self.online_batch_size, self.num_grad_steps = int(training_mode.split(',')[1]), \
                                                int(training_mode.split(',')[2])
            self.reset = 'reset' in training_mode
        

        self.true_online = true_online


        self.monitor = 'monitor' in elbo_mode


        if not self.training_mode == 'closed_form':
            self.elbo_options = {}
            if 'score' in elbo_mode: 
                for option in ['paris', 'variance_reduction', 'mcmc']:
                    self.elbo_options[option] = True if option in elbo_mode else False

                if 'bptt_depth' in elbo_mode: 
                    self.elbo_options['bptt_depth'] = int(elbo_mode.split('bptt_depth')[1].split('_')[1])

                self.elbo_options['true_online'] = True if self.true_online else False
        
        def optimizer_update_fn(params, updates):
            new_params = optax.apply_updates(params, updates)
            return new_params
        
        self.optimizer_update_fn = optimizer_update_fn


        self.trainable_params = tree_map(lambda x: x == '', self.frozen_params)

        schedule = None

        if 'linear_sched' in optim_options:
            
            learning_rate = optax.linear_schedule(learning_rate, 
                                                  end_value=10*learning_rate, 
                                                  transition_begin=2000,
                                                  transition_steps=num_epochs * (seq_length / self.online_batch_size))
        
        elif 'warmup_cosine' in optim_options:
            print('Setting up warmup cosine schedule. Starting at {} and ending at {}.'.format(learning_rate / 10, 
                                                                                               learning_rate))

            learning_rate = optax.warmup_cosine_decay_schedule(
                                    init_value=learning_rate / 10,
                                    peak_value=learning_rate,
                                    warmup_steps=10000,
                                    decay_steps=seq_length,
                                    end_value=learning_rate)
            
        elif 'gamma' in optim_options:
            gamma = float(optim_options.split(',')[-1].split('_')[1])
            schedule = optax.scale_by_schedule(lambda t: (t+1)**(-gamma))

        
        else:
            pass

        
        base_optimizer = optax.apply_if_finite(getattr(optax, optimizer)(learning_rate),
                                            max_consecutive_errors=10)
        
        
        if schedule is not None:
            optimizer = optax.chain(base_optimizer, schedule)
        else: 
            optimizer = base_optimizer

        self.optimizer = optimizer

        if 'closed_form' in self.elbo_mode: 
            print('USING ANALYTICAL ELBO.')
            self.elbo = LinearGaussianELBO(self.p, self.q)
            def elbo_and_grads_batch(key, ys, params):
                def f(params):
                    elbo, aux = self.elbo(ys, 
                                        len(ys)-1, 
                                        self.formatted_theta_star, 
                                        q.format_params(params))
                    return -elbo / len(ys), aux 
                (neg_elbo, aux), neg_grad = jax.value_and_grad(f, has_aux=True)(params)
                return (-neg_elbo, neg_grad), aux
            self.elbo_step = None
            
        elif 'score' in self.elbo_mode:
            print('USING SCORE ELBO.')
            if 'truncated' in self.elbo_mode: 
                print('Using truncated gradients.')
                self.elbo = OnlineELBOScoreTruncatedGradients(
                                                self.p, 
                                                self.q, 
                                                num_samples=num_samples, 
                                                **self.elbo_options)
            else: 
                print('Using full gradients.')
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
                elbo = elbo / (T+1)
                neg_grad = tree_map(lambda x: -x / (T+1), grad)
                return (elbo, neg_grad), aux
                    
            self.elbo_step = self.elbo.step
            
                
        elif ('autodiff_on_backward' in self.elbo_mode) and self.reset:
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
                return (-neg_elbo, neg_grad), aux

        else:
            print('ELBO mode not suitable for gradient accumulation.')
            raise NotImplementedError

        self.elbo_batch = elbo_and_grads_batch
        self.get_montecarlo_keys = get_keys

        if self.reset:
            init_carry = 0.0
        else: 
            params = self.q.get_random_params(jax.random.PRNGKey(0))
            params = self._extract_params(params)
            init_carry = self.elbo.init_carry(self._build_params(params))

        self.init_carry = init_carry

        def update(key, 
                elbo_carry, 
                strided_ys, 
                timesteps, 
                params):
            
            if self.reset:
                
                (elbo, neg_grad), aux = self.elbo_batch(key, strided_ys, params)

            else: 

                if self.true_online: 
                    t = timesteps[0]
                    strided_ys = strided_ys[0]
                    def _step(carry, x):
                        key, t, strided_y = x

                        input_t = {'t':t, 
                                'key': key, 
                                'ys_bptt':strided_y, 
                                'T':t,
                                'phi':params}
                        
                        carry['theta'] = self.formatted_theta_star
                        new_carry, aux = self.elbo.step(carry, 
                                                    input_t)
                        return new_carry, aux

                    new_carry, aux = _step(elbo_carry, (key, t, strided_ys))
                    elbo_t, grad_t = self.elbo.postprocess(new_carry)
                    if self.online_difference:
                        grad_tm1 = self.elbo.postprocess(elbo_carry)[1]
                        neg_grad = tree_map(lambda x,y: -(x-y), grad_t, grad_tm1)
                    else: 
                        neg_grad = tree_map(lambda x: -x, grad_t)

                    elbo = elbo_t / (t+1)
                    elbo_carry = new_carry
                else: 
                    keys = jax.random.split(key, len(timesteps))

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
                    elbo/=(T+1)
                    neg_grad = tree_map(lambda x: -x / (T+1), grad)

        
            return elbo, neg_grad, elbo_carry, aux
                    
        
        self.update = update

    def timesteps(self, seq_length, key):
        all_timesteps = jnp.arange(0, seq_length)
        if key is None: 
            cnts = range(0, seq_length, self.online_batch_size)
        else: 
            cnts = jax.random.permutation(key, jnp.arange(0, seq_length, self.online_batch_size))
        
        for cnt in cnts:
            yield all_timesteps[cnt:cnt+self.online_batch_size]

    def fit(self, 
            key_params, 
            key_montecarlo, 
            data, 
            log_writer=None, 
            args=None, 
            log_writer_monitor=None):

        params = self.q.get_random_params(key_params, args)
        params = self._extract_params(params)
        
        
        opt_state = self.optimizer.init(params)
        xs, ys = data
        xs = xs[0]
        ys = ys[0]
        seq_length = ys.shape[0]
        keys = get_keys(key_montecarlo, 
                        seq_length // self.online_batch_size, 
                        self.num_epochs)
        
        @jax.jit
        def step(key, strided_ys_on_timesteps, ys_on_timesteps, elbo_carry, timesteps, params, opt_state):
            
            if self.true_online:
                opt_state = self.optimizer.init(params)
#
            if self.monitor:
                monitor_elbo_value = self.monitor_elbo(key, 
                                                       ys_on_timesteps, 
                                                       len(ys_on_timesteps)-1, 
                                                       self.formatted_theta_star, 
                                                       self.q.format_params(self._build_params(params)))[0] / len(ys_on_timesteps)
            else:
                monitor_elbo_value = None
            


            def inner_step(carry, x):
                inner_carry, params, opt_state = carry
                key = x
                elbo, neg_grad, new_carry, aux = self.update(
                                                            key, 
                                                            inner_carry,
                                                            strided_ys_on_timesteps, 
                                                            timesteps, 
                                                            self._build_params(params))
                neg_grad = self._extract_params(neg_grad)

                updates, opt_state = self.optimizer.update(neg_grad, 
                                                            opt_state, 
                                                            params)
                
                new_params = self.optimizer_update_fn(params, updates)
                
                return (lax.cond(timesteps[-1] == 0, 
                                 lambda x:x, 
                                 lambda x:elbo_carry, 
                                 new_carry), 
                        new_params, 
                        opt_state), \
                        (elbo, aux, new_carry)
            
            (_, new_params, opt_state), (elbos, aux, inner_steps_carries) = jax.lax.scan(inner_step, 
                                                        init=(elbo_carry, params, opt_state), 
                                                        xs=jax.random.split(key, 
                                                                            self.num_grad_steps))


            elbo_carry = tree_get_idx(-1, inner_steps_carries)

            if self.true_online:
                params_q_t, params_q_tm1_t = tree_get_idx(-1, aux)
                aux = self.q.smoothing_means_tm1_t(params_q_t, params_q_tm1_t, 10000, key)
            else: 
                aux = None

            if not isinstance(self.q, NonAmortizedBackwardSmoother):
                params = new_params
            return (params, opt_state, elbo_carry), (elbos, aux, monitor_elbo_value)
        

        absolute_step_nb = 0

        # jitted_step = jax.jit(step)

        for epoch_nb, keys_epoch in enumerate(keys):
            elbo_carry = self.init_carry
            strided_ys = self.elbo.preprocess(ys)

            timesteps_lists = self.timesteps(seq_length, None)

            
            for step_nb, (timesteps, key_step) in enumerate(zip(timesteps_lists, keys_epoch)):
                # if step_nb > 1: 
                #     with open('jitted_step.txt', 'w') as f: 
                #         f.write(jitted_step.lower(key_step, 
                #                         strided_ys[timesteps], 
                #                         ys[timesteps],
                #                         elbo_carry, 
                #                         timesteps, 
                #                         params, 
                #                         opt_state).compile().as_text())
                    
                (params, opt_state, elbo_carry), (elbos, aux, monitor_elbo) = step(
                                                                                key_step, 
                                                                                strided_ys[timesteps], 
                                                                                ys[timesteps],
                                                                                elbo_carry, 
                                                                                timesteps, 
                                                                                params, 
                                                                                opt_state)
                


                


                if self.true_online: 
                    t = timesteps[-1]
                    x_t = xs[t]
                    if t > 0: 
                        x_tm1 = xs[t-1]
                    with log_writer.as_default():
                        tf.summary.scalar('Filtering RMSE', 
                                            jnp.sqrt(jnp.mean((x_t - aux[1])**2)),
                                            absolute_step_nb)

                        if t > 0:
                            tf.summary.scalar('1-step smoothing RMSE', 
                                            jnp.sqrt(jnp.mean((x_tm1 - aux[0])**2)),
                                            absolute_step_nb)

                with log_writer.as_default():
                    for inner_step_nb, elbo in enumerate(elbos): 
                        tf.summary.scalar('ELBO', elbo, self.num_grad_steps*absolute_step_nb + inner_step_nb)

                    

                if self.monitor: 
                    with log_writer_monitor.as_default():
                        tf.summary.scalar('ELBO', monitor_elbo, absolute_step_nb)

                absolute_step_nb += 1 

                yield params, elbo, aux

    def multi_fit(self, 
                  key, 
                  data, 
                  num_fits, 
                  log_dir='', 
                  args=None):

        print('Starting training...')
        
        tensorboard_subdir = os.path.join(log_dir, 'tensorboard_logs')
        os.makedirs(tensorboard_subdir, exist_ok=True)
        times = []
        def run_fit(fit_nb, fit_key):

            log_writer = tf.summary.create_file_writer(os.path.join(tensorboard_subdir, f'fit_{fit_nb}'))
            if self.monitor:
                log_writer_monitor = tf.summary.create_file_writer(os.path.join(tensorboard_subdir, f'fit_{fit_nb}_monitor'))
            else:
                log_writer_monitor = None
            print(f'Starting fit {fit_nb}/{num_fits-1}...')

            key_params, key_montecarlo = jax.random.split(fit_key, 2)

            params, elbos, means_t, means_tm1 = [], [], [], []

            time0 = time()
            for global_step_nb, (params_at_step, elbo_at_step, aux_at_step) in enumerate(self.fit(key_params, 
                                                                                    key_montecarlo, 
                                                                                    data, 
                                                                                    log_writer, 
                                                                                    args, 
                                                                                    log_writer_monitor)):
                params.append(params_at_step)
                elbos.append(elbo_at_step)
                if self.true_online:
                    means_tm1.append(aux_at_step[0])
                    means_t.append(aux_at_step[1])
                times.append(time() - time0)
                time0 = time()
                

            burnin = int(0.75*self.num_epochs)
            best_step_for_fit = burnin + jnp.nanargmax(jnp.array(elbos[burnin:]))


            if self.true_online:
                jnp.save(os.path.join(log_dir, 'x_tm1.npy'), jnp.array(means_tm1)[1:])
                jnp.save(os.path.join(log_dir, 'x_t.npy'), jnp.array(means_t))

            best_elbo_for_fit = elbos[best_step_for_fit]
            best_params_for_fit = params[best_step_for_fit]
            print(f'Fit {fit_nb}: best ELBO {best_elbo_for_fit:.3f} at step {best_step_for_fit}')
            return best_params_for_fit, best_elbo_for_fit, jnp.array(times[20:])
        
        fit_nbs = range(args.num_fits)
        fit_keys = jax.random.split(key, args.num_fits)
        best_params_per_fit, best_elbos_per_fit = [], []

        all_timings = []
        for fit_nb, fit_key in zip(fit_nbs, fit_keys): 

            best_params_for_fit, best_elbo_for_fit, timings = run_fit(fit_nb, fit_key)
            best_params_per_fit.append(best_params_for_fit)
            best_elbos_per_fit.append(best_elbo_for_fit)
            all_timings.append(timings)


        best_optim = jnp.argmax(jnp.array(best_elbos_per_fit))
        print(f'Best fit is {best_optim}.')
        training_info = dict()
        training_info['avg_time'] = jnp.mean(jnp.concatenate(all_timings)).tolist()
        training_info['best_fit'] = best_optim.tolist()
        save_dict(training_info, 'training_info', log_dir)

        return best_params_per_fit[best_optim], best_params_per_fit

