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
                self.elbo = OnlineELBOScoreTruncatedGradients(
                                                self.p, 
                                                self.q, 
                                                num_samples=num_samples, 
                                                **self.elbo_options)
            else: 
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
            
            (_, params, opt_state), (elbos, aux, inner_steps_carries) = jax.lax.scan(inner_step, 
                                                        init=(elbo_carry, params, opt_state), 
                                                        xs=jax.random.split(key, 
                                                                            self.num_grad_steps))


            elbo_carry = tree_get_idx(-1, inner_steps_carries)
            params_q_t, params_q_tm1_t = tree_get_idx(-1, aux)
            means_tm1, means_t = self.q.smoothing_means_tm1_t(params_q_t, params_q_tm1_t, 10000, key)
            return (params, opt_state, elbo_carry), (elbos, (means_tm1, means_t), monitor_elbo_value)
        

        absolute_step_nb = 0
        dummy_filt_mean_and_cov = jnp.empty((self.p.state_dim,)), jnp.empty((self.p.state_dim, self.p.state_dim))
        zeros = jnp.zeros((self.p.state_dim, self.p.state_dim))


        @partial(jax.jit, static_argnums=(0,))
        def get_bivariate_joint_for_linear_backwd_smoother(model:BackwardSmoother, filt_stats, params):
            params = model.format_params(params)
            backwd_params = LinearBackwardSmoother.\
                                linear_gaussian_backwd_params_from_transition_and_filt(filt_stats[0].mean, filt_stats[0].scale.cov, 
                                                                                    params.transition)
            A_backwd, a_backwd, Sigma_backwd = \
                    backwd_params.map.w, backwd_params.map.b, backwd_params.noise.scale.cov
            smoothing_dist = Gaussian.Params(
                                            mean=A_backwd @ filt_stats[1].mean + a_backwd, 
                                            scale=Scale(cov=Sigma_backwd + \
                                                        A_backwd @ filt_stats[1].scale.cov @ A_backwd.T))
            bivariate_joint = Gaussian.Params(mean=jnp.concatenate([filt_stats[1].mean, 
                                                                    smoothing_dist.mean]),
                                            scale=Scale(cov=jnp.block([[filt_stats[1].scale.cov, zeros],
                                                                        [zeros, smoothing_dist.scale.cov]])))  
            
            return smoothing_dist, bivariate_joint
        

        # init_params = params
        for epoch_nb, keys_epoch in enumerate(keys):
            elbo_carry = self.init_carry
            strided_ys = self.elbo.preprocess(ys)

            timesteps_lists = self.timesteps(seq_length, None)
            # logl_carry = (*dummy_filt_mean_and_cov, 0.0)

            # filt_stats_true = [Gaussian.Params(mean=dummy_filt_mean_and_cov[0], 
            #                                    scale=Scale(cov=dummy_filt_mean_and_cov[1])), None]
            
            # filt_stats_var = [Gaussian.Params(mean=dummy_filt_mean_and_cov[0], 
            #                                   scale=Scale(cov=dummy_filt_mean_and_cov[1])), None]
            
            for step_nb, (timesteps, key_step) in enumerate(zip(timesteps_lists, keys_epoch)):

                (new_params, opt_state, elbo_carry), (elbos, (means_tm1, means_t), monitor_elbo) = step(
                                                                                            key_step, 
                                                                                            strided_ys[timesteps], 
                                                                                            ys[timesteps],
                                                                                            elbo_carry, 
                                                                                            timesteps, 
                                                                                            params, 
                                                                                            opt_state)
                if not ('nonamortized' in self.elbo_mode):
                    params = new_params
                
                if self.true_online: 
                    t = timesteps[-1]
                    x_t = xs[t]
                    if t > 0: 
                        x_tm1 = xs[t-1]

                # if isinstance(self.p, LinearGaussianHMM) and self.true_online:
                #     logl_carry, logl = Kalman.recursive_logl_step(timesteps, 
                #                                                   ys[timesteps], 
                #                                                   logl_carry, 
                #                                                   self.formatted_theta_star)
                #     filt_stats_true[1] = Gaussian.Params(mean=logl_carry[0], 
                #                                          scale=Scale(cov=logl_carry[1]))
                    
                #     x_t = filt_stats_true[1].mean

                #     x_tm1 = get_bivariate_joint_for_linear_backwd_smoother(self.p, 
                #                                                             filt_stats_true, 
                #                                                             self.theta_star)[0].mean
                    

                if self.true_online:
                    
                        # variational_filt_dist_params = tree_get_idx(inner_step_nb, aux)
                        # inner_step_params = self._build_params(tree_get_idx(inner_step_nb, inner_steps_params))
                        # filt_stats_var[1] = variational_filt_dist_params
                        # smoothing_tm1, bivariate_joint = get_bivariate_joint_for_linear_backwd_smoother(
                        #                                                         self.q, 
                        #                                                         filt_stats_var, 
                        #                                                         inner_step_params)
                    with log_writer.as_default():

                        tf.summary.scalar('Filtering RMSE', 
                                            jnp.sqrt(jnp.mean((x_t - means_t)**2)),
                                            absolute_step_nb)

                        if t > 0:
                            tf.summary.scalar('1-step smoothing RMSE', 
                                            jnp.sqrt(jnp.mean((x_tm1 - means_tm1)**2)),
                                            absolute_step_nb)
                                
                                # tf.summary.scalar('Bivariate smoothing RMSE', 
                                #                 jnp.sqrt(jnp.mean((jnp.concatenate([x_tm1, x_t]) - bivariate_joint.mean)**2)),
                                #                 self.num_grad_steps*absolute_step_nb + inner_step_nb)


                
                #     with log_writer.as_default():
                #         for inner_step_nb, elbo in enumerate(elbos): 
                #             tf.summary.scalar('true logl', logl / (timesteps[-1] + 1), 
                #                               self.num_grad_steps*absolute_step_nb + inner_step_nb)
                        


                #                 tf.summary.scalar('filt KL', 
                #                                   Gaussian.KL(filt_stats_var[1], filt_stats_true[1]), 
                #                                   self.num_grad_steps*absolute_step_nb + inner_step_nb)                            
                                
                                
                #                 if timesteps[-1] > 0:
                #                     tf.summary.scalar('bivariate joint KL', 
                #                                       Gaussian.KL(variational_joints[-1], true_joint), 
                #                                       self.num_grad_steps*absolute_step_nb + inner_step_nb)                            
                    # filt_stats_true = [filt_stats_true[1], None]
                    # filt_stats_var = [filt_stats_var[1], None]


                elif isinstance(self.p, LinearGaussianHMM) and (not self.true_online):
                    logl = self.p.likelihood_seq(ys[timesteps], self.theta_star)
                    with log_writer.as_default():
                        tf.summary.scalar('true logl', logl / (timesteps[-1] + 1), absolute_step_nb)
                        
                with log_writer.as_default():
                    for inner_step_nb, elbo in enumerate(elbos): 
                        tf.summary.scalar('ELBO', elbo, self.num_grad_steps*absolute_step_nb + inner_step_nb)

                    

                if self.monitor: 
                    with log_writer_monitor.as_default():
                        tf.summary.scalar('ELBO', monitor_elbo, absolute_step_nb)

                absolute_step_nb += 1 

                yield params, elbo

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

            params, elbos = [], []
            # intermediate_params_dir = os.path.join(log_dir, f'fit_{fit_nb}_intermediate_params')
            # os.makedirs(intermediate_params_dir, exist_ok=True)
            time0 = time()
            for global_step_nb, (params_at_step, elbo_at_step) in enumerate(self.fit(key_params, 
                                                                                    key_montecarlo, 
                                                                                    data, 
                                                                                    log_writer, 
                                                                                    args, 
                                                                                    log_writer_monitor)):
                # if global_step_nb % 10 == 0:
                #     save_params(params_at_step, f'phi_{global_step_nb}', intermediate_params_dir)
                params.append(params_at_step)
                elbos.append(elbo_at_step)
                times.append(time() - time0)
                time0 = time()
                

            burnin = 100
            if self.true_online: 
                best_step_for_fit = burnin + jnp.nanargmax(jnp.array(elbos[burnin:]))
            else: 
                best_step_for_fit = jnp.nanargmax(jnp.array(elbos))
                
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

        return best_params_per_fit[best_optim], best_params_per_fit, 

