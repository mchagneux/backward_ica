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
from jax_tqdm import scan_tqdm

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
                seq_length,
                num_samples=1, 
                force_full_mc=False,
                frozen_params='',
                num_seqs=1,
                training_mode='offline',
                elbo_mode='autodiff_on_backward',
                logging_type='basic_logging'):
        
        self.num_epochs = num_epochs
        self.q = q 
        self.seq_length = seq_length
        self.q.print_num_params()
        self.p = p
        self.theta_star = theta_star
        self.formatted_theta_star = self.p.format_params(theta_star, 
                                                         precompute=['prec'])
        self.frozen_params = frozen_params

        self.num_seqs = num_seqs
        self.logging_type = logging_type
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
            if 'recompute' in training_mode:
                self.online_batch_size = 2
            else: 
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
                for option in ['paris', 'mcmc']:
                    self.elbo_options[option] = True if option in elbo_mode else False

                if 'bptt_depth' in elbo_mode: 
                    self.elbo_options['bptt_depth'] = int(elbo_mode.split('bptt_depth')[1].split('_')[1])
                
                if 'unnormalized' in elbo_mode:
                    self.elbo_options['normalizer'] = lambda x: jnp.exp(x) / x
                else:
                    self.elbo_options['normalizer'] = exp_and_normalize

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


        if isinstance(self.q, ConjugateForward):
            self.elbo = GeneralForwardELBO(self.p, 
                                           self.q, 
                                           num_samples)
            print('USING AUTODIFF ON FORWARD ELBO.')

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

        elif 'closed_form' in self.elbo_mode: 
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
            init_carry = jnp.zeros((self.num_seqs,))
        else: 
            params = self.q.get_random_params(jax.random.PRNGKey(0))
            params = self._extract_params(params)
            init_carry = jax.vmap(self.elbo.init_carry, 
                                  in_axes=None, 
                                  length=self.num_seqs)(self._build_params(params))

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
                        return new_carry, (new_carry, aux)
                    
                    if 'recompute' in self.training_mode:
                        keys = jax.random.split(key, len(timesteps))

                        _ , (carries, aux) = jax.lax.scan(_step, 
                                                          init=elbo_carry, 
                                                          xs=(keys, timesteps, strided_ys),
                                                          unroll=len(timesteps))
                        elbo_carry = tree_get_idx(0, carries)
                        new_carry = tree_get_idx(-1, carries)
                    else: 
                        new_carry, (_, aux) = _step(elbo_carry, (key, timesteps[-1], strided_ys[-1]))
                        
                    elbo_t, grad_t = self.elbo.postprocess(new_carry)
                    if self.online_difference:
                        grad_tm1 = self.elbo.postprocess(elbo_carry)[1]
                        neg_grad = tree_map(lambda x,y: -(x-y), grad_t, grad_tm1)
                    else: 
                        neg_grad = tree_map(lambda x: -x / (timesteps[-1] + 1), grad_t)

                    elbo = elbo_t / (timesteps[-1]+1)
                    if not 'recompute' in self.training_mode:
                        elbo_carry = new_carry
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
                    elbo/=(T+1)
                    neg_grad = tree_map(lambda x: -x / (T+1), grad)

        
            return elbo, neg_grad, elbo_carry, aux
                    
        
        self.update = update

    def timesteps(self, seq_length, key):

        all_timesteps = jnp.arange(0, seq_length)
        if key is None: 
            if self.true_online:
                cnts = range(0, seq_length, 1)
            else:
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
        seq_length = self.seq_length
        keys = get_keys(key_montecarlo, 
                        seq_length // self.online_batch_size, 
                        self.num_epochs)
        
        strided_ys = jax.vmap(self.elbo.preprocess)(ys)

        def _step(step_carry, x):
            params, opt_state, elbo_carry = step_carry
            key, timesteps = x 
            strided_ys_on_timesteps = strided_ys[:,timesteps]
            
            if self.true_online and (not self.online_difference):
                opt_state = self.optimizer.init(params)

            def inner_step(carry, x):
                inner_carry, params, opt_state = carry
                key = x
                
                elbo, neg_grad, new_carry, aux = jax.vmap(self.update, in_axes=(0, 0, 0, None, None))(
                                                            jax.random.split(key, len(strided_ys_on_timesteps)), 
                                                            lax.cond(timesteps[-1] > 0, 
                                                                     lambda x:x, 
                                                                     lambda x:elbo_carry,
                                                                     inner_carry),
                                                            strided_ys_on_timesteps, 
                                                            timesteps, 
                                                            self._build_params(params))
                neg_grad = self._extract_params(neg_grad)

                neg_grad = tree_map(partial(jnp.mean, axis=0), neg_grad)

                updates, opt_state = self.optimizer.update(neg_grad, 
                                                            opt_state, 
                                                            params)
                
                new_params = self.optimizer_update_fn(params, updates)
                
                return (new_carry, 
                        new_params, 
                        opt_state), \
                        (jnp.mean(elbo, axis=0), aux)
            
            
            (elbo_carry, params, opt_state), (elbos, aux) = jax.lax.scan(inner_step, 
                                                        init=(elbo_carry, params, opt_state), 
                                                        xs=jax.random.split(key, 
                                                                            self.num_grad_steps))
            
            if self.num_grad_steps > 1:
                params_q_t, params_q_tm1_t = tree_get_idx(-1, tree_get_idx(-1, aux))
                aux = self.q.smoothing_means_tm1_t(params_q_t, params_q_tm1_t, 10000, key)
            else: 
                aux = None

            # if not isinstance(self.q, NonAmortizedBackwardSmoother):
            return (params, opt_state, elbo_carry), (elbos, aux)
        
        absolute_step_nb = 0


        if self.monitor: 
            monitor_elbo = jax.jit(lambda key, ys, T, formatted_theta_star, phi:self.monitor_elbo(
                                                                                    key, 
                                                                                    ys, 
                                                                                    T, 
                                                                                    formatted_theta_star, 
                                                                                    self.q.format_params(phi))[0] / len(ys))
        else: 
            monitor_elbo = lambda *args: 0.0
            
        elbo_carry = self.init_carry

        if self.logging_type == 'basic_logging':

            if self.true_online:
                all_timesteps = jnp.expand_dims(jnp.arange(0, seq_length), 
                                                    axis=-1)
            else: 
                all_timesteps = jnp.array(list(self.timesteps(seq_length, None)))

            if self.num_epochs == 1:
                @scan_tqdm(len(all_timesteps))
                def step(carry, x):
                    return _step(carry, x[1:])
            else: 
                def step(carry, x):
                    return _step(carry, x[1:])
            


            @scan_tqdm(self.num_epochs)
            def _epoch_step(carry, x):
                params, opt_state, elbo_carry = carry
                _, keys_epoch = x
                (params, opt_state, elbo_carry), (elbos_steps, _) = jax.lax.scan(step, 
                                                                init=(params, opt_state, elbo_carry),
                                                                xs=(jnp.arange(0, len(all_timesteps)),
                                                                    keys_epoch, 
                                                                    all_timesteps))

                return (params, opt_state, elbo_carry), elbos_steps[-1][-1]
                
            (params, _, elbo_carry), elbos_epochs = jax.lax.scan(_epoch_step, 
                                                 init=(params, opt_state, elbo_carry), 
                                                 xs = (jnp.arange(0, self.num_epochs), keys))
            
            return params, elbos_epochs[-1] #self.elbo.postprocess(elbo_carry)[0] / seq_length
                                                                            
                        
        else: 
            all_params, all_elbos, all_means_tm1, all_means_t = dict(), dict(), dict(), dict()
            jitted_step = jax.jit(_step)


            for epoch_nb, keys_epoch in enumerate(keys): 
                timesteps_lists = self.timesteps(seq_length, None)
                for _ , (timesteps, key_step) in enumerate(zip(timesteps_lists, 
                                                                    keys_epoch)):
                    carry = params, opt_state, elbo_carry
                    x = key_step, timesteps
                    (params, opt_state, elbo_carry), (elbos, aux) = jitted_step(carry, x)
        
                    if log_writer is not None:
                        if self.num_grad_steps > 1: 
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

                        if self.online_batch_size != seq_length:
                            with log_writer.as_default():
                                for inner_step_nb, elbo in enumerate(elbos): 
                                    tf.summary.scalar('ELBO at inner step', elbo, self.num_grad_steps*absolute_step_nb + inner_step_nb)
                    if absolute_step_nb % 100 == 0:
                        all_params[absolute_step_nb] = params
                        all_elbos[absolute_step_nb] = elbos[-1] 
                    if 'truncated' in self.elbo_mode:
                        all_means_tm1[absolute_step_nb] = aux[0]
                        all_means_t[absolute_step_nb] = aux[1]
                    absolute_step_nb += 1

                if self.monitor: 
                    monitor_elbo_value = monitor_elbo(key_step, ys, len(ys)-1, self.formatted_theta_star, params)

                    with log_writer.as_default():
                        tf.summary.scalar('Unbiased ELBO at epoch', monitor_elbo_value, epoch_nb)
                if log_writer is not None:
                    with log_writer.as_default():
                        tf.summary.scalar('ELBO at epoch', elbos[-1], epoch_nb)

                return all_params, all_elbos, (all_means_tm1, all_means_t)

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

            if self.logging_type == 'tensorboard':
                log_writer = tf.summary.create_file_writer(os.path.join(tensorboard_subdir, f'fit_{fit_nb}'))
                if self.monitor:
                    log_writer_monitor = tf.summary.create_file_writer(os.path.join(tensorboard_subdir, f'fit_{fit_nb}_monitor'))
                else:
                    log_writer_monitor = None
            else:
                log_writer = None 
                log_writer_monitor = None
            print(f'Starting fit {fit_nb}/{num_fits-1}...')

            key_params, key_montecarlo = jax.random.split(fit_key, 2)


            # time0 = time()
            if self.logging_type == 'tensorboard':
                params, elbos, (means_tm1, means_t) = self.fit(
                                                            key_params, 
                                                            key_montecarlo, 
                                                            data, 
                                                            log_writer, 
                                                            args, 
                                                            log_writer_monitor)


                print(elbos)

                burnin = int(0.75*self.num_epochs)

                elbos = {k:v for k,v in elbos.items() if k > burnin}

                best_step_for_fit = max(elbos, key=elbos.get)

                if 'truncated' in self.elbo_mode:
                    jnp.save(os.path.join(log_dir, 'x_tm1.npy'), jnp.array(means_tm1)[1:])
                    jnp.save(os.path.join(log_dir, 'x_t.npy'), jnp.array(means_t))

                best_elbo_for_fit = elbos[best_step_for_fit]
                best_params_for_fit = params[best_step_for_fit]
                print(f'Fit {fit_nb}: best ELBO {best_elbo_for_fit:.3f} at step {best_step_for_fit}')
                return best_params_for_fit, best_elbo_for_fit, jnp.array(times[20:])
        

            else: 
                final_params, final_elbo = self.fit(key_params, 
                                key_montecarlo, 
                                data, 
                                log_writer, 
                                args, 
                                log_writer_monitor)
                print(f'Fit {fit_nb}: final ELBO {final_elbo:.3f}')
                return final_params, final_elbo, jnp.array(times[20:])
                # time0 = time()
                

        
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

