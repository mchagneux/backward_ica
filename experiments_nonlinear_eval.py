import argparse
import jax 
from backward_ica import hmm 
from backward_ica import utils
import matplotlib.pyplot as plt 
import pickle
utils.enable_x64(True)
import os 
import subprocess



def run_ffbsi_em_on_train_data(train_args, mle_args):

    key_theta = jax.random.PRNGKey(train_args.seed_theta)

    utils.set_global_cov_mode(train_args)

    key_params, key_gen, key_mle = jax.random.split(key_theta, 3)

    p = hmm.NonLinearGaussianHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                            layers=train_args.emission_map_layers,
                            slope=train_args.slope,
                            num_particles=mle_args.num_particles,
                            transition_bias=train_args.transition_bias,
                            injective=train_args.injective) # specify the structure of the true model
    
    theta_star = p.get_random_params(key_params)
    utils.save_params(theta_star, 'theta', train_args.save_dir)

    obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, train_args.num_seqs, train_args.seq_length)[1]
    
    print('Starting FFBSi EM training...')
    theta_mle, logls_mle = p.fit_ffbsi_em(key_mle, 
                                        obs_seqs, 
                                        optimizer=mle_args.optimizer, 
                                        learning_rate=mle_args.learning_rate, 
                                        batch_size=mle_args.batch_size, 
                                        num_epochs=mle_args.num_epochs)

    utils.save_params(theta_mle, 'phi', mle_args.save_dir)

    utils.save_train_logs((0, None, [logls_mle]), mle_args.save_dir, plot=False)

def run_ffbsi_smoothing_on_eval_data(train_args, eval_args, ground_truth=True):

    utils.set_global_cov_mode(train_args)

    key = jax.random.PRNGKey(eval_args.seed)

    p = hmm.NonLinearGaussianHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                            layers=train_args.emission_map_layers,
                            slope=train_args.slope,
                            num_particles=eval_args.num_particles,
                            transition_bias=train_args.transition_bias,
                            injective=train_args.injective) # specify the structure of the true model

    theta_star = utils.load_params('theta', train_args.save_dir)
    key_gen, key_ffbsi = jax.random.split(key,2)
    obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, eval_args.num_seqs, eval_args.seq_length)[1]
    timesteps = range(2, eval_args.seq_length, eval_args.step)
    
    print(obs_seqs[0][list(timesteps)[0]])

    if ground_truth:
        print('Starting FFBSi smoothing with ground truth parameters...')
        theta = theta_star
    
    else: 
        print('Starting FFBSi smoothing with MLE parameters...')
        theta = utils.load_params('phi', eval_args.path_phi)

    smoothing_ffbsi = utils.multiple_length_ffbsi_smoothing(key_ffbsi, obs_seqs, p, theta, timesteps)

    with open(os.path.join(eval_args.save_dir, 'smoothing_results'), 'wb') as f:
        pickle.dump(smoothing_ffbsi, f)

def eval(exp_date, exp_name, q_versions_and_dates, run=False):
    
    base_dir = os.path.join('experiments', 'p_nonlinear', exp_name)


    train_dirs = []
    eval_dirs = []

    for (version, date) in q_versions_and_dates:
        train_dir = os.path.join(base_dir, 'trainings', version, date)
        eval_dir = os.path.join(base_dir, 'evals', 'results', f'{version}_{date}', exp_date)
        os.makedirs(eval_dir, exist_ok=True)
        train_dirs.append(train_dir)
        eval_dirs.append(eval_dir)

    num_eval_seqs = 5

    eval_seqs_length = 500
    eval_step = 50
    eval_seed = 0

    train_args = utils.load_args('train_args', train_dirs[0])
    mle_name = 'ffbsi_em'
    mle_save_dir = os.path.join(base_dir, 'trainings', mle_name, exp_date)
    mle_args = argparse.Namespace()
    mle_args.num_particles = 20
    mle_args.learning_rate = 1e-1
    mle_args.optimizer = train_args.optimizer
    mle_args.batch_size = train_args.num_seqs
    mle_args.num_epochs = train_args.num_epochs
    mle_args.q_version = 'mle'
    mle_args.save_dir = mle_save_dir

    # train_dirs += [mle_save_dir]
    
    if run:
        processes = [subprocess.Popen(f'python eval_nonlinear.py \
                                    --train_dir {train_dir} \
                                    --save_dir {eval_dir} \
                                    --step {eval_step} \
                                    --num_seqs {num_eval_seqs} \
                                    --seq_length {eval_seqs_length} \
                                    --seed {eval_seed}', 
                                shell=True) \
                        for (train_dir, eval_dir) in zip(train_dirs, eval_dirs)]

        for process in processes:
            process.wait()

    # os.makedirs(mle_save_dir, exist_ok=True)

    # utils.save_args(mle_args, 'train_args', mle_save_dir)   

    # run_ffbsi_em_on_train_data(train_args, mle_args)

    ffbsi_smoothing_save_dir = os.path.join(base_dir, 'evals', 'results', mle_name, exp_date)
    # os.makedirs(ffbsi_smoothing_save_dir, exist_ok=True)
    eval_args = utils.load_args('eval_args', eval_dirs[0])
    eval_args_ffbsi_smoothing = argparse.Namespace()
    eval_args_ffbsi_smoothing.save_dir = ffbsi_smoothing_save_dir
    eval_args_ffbsi_smoothing.num_seqs = eval_args.num_seqs 
    eval_args_ffbsi_smoothing.seq_length = eval_args.seq_length
    eval_args_ffbsi_smoothing.seed = eval_args.seed
    eval_args_ffbsi_smoothing.step = eval_args.step
    eval_args_ffbsi_smoothing.num_particles = 1000
    #eval_dirs += [ffbsi_smoothing_save_dir]

    eval_args_ffbsi_smoothing.path_phi = mle_save_dir

    # utils.save_args(eval_args_ffbsi_smoothing, 'eval_args', ffbsi_smoothing_save_dir)
    # run_ffbsi_smoothing_on_eval_data(train_args, eval_args_ffbsi_smoothing, ground_truth=False)
    method_names = [f'{q_version_and_date[0]}_{q_version_and_date[1]}' for q_version_and_date in q_versions_and_dates] #+ [f'{mle_name}_{date}']

    ffbsi_smoothing_save_dir = os.path.join(base_dir, 'evals', 'results', 'ffbsi_gt', exp_date)

    if run:
        os.makedirs(ffbsi_smoothing_save_dir, exist_ok=True)
        eval_args_ffbsi_smoothing.save_dir = ffbsi_smoothing_save_dir
        utils.save_args(eval_args_ffbsi_smoothing, 'eval_args', ffbsi_smoothing_save_dir)
        run_ffbsi_smoothing_on_eval_data(train_args, eval_args_ffbsi_smoothing, ground_truth=True)
    ref_dir = ffbsi_smoothing_save_dir


    save_dir = os.path.join(base_dir, 'evals','comparisons', exp_date)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'method_names.txt'), 'w') as f: 
        f.write(str(method_names))

    utils.compare_multiple_length_smoothing(ref_dir, eval_dirs, train_dirs, method_names, save_dir)


if __name__ == '__main__': 
    from datetime import datetime
    exp_date = '2022_05_19__16_24_08' #datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    exp_name = 'p_noninjective_2'
    q_versions_and_dates = [#('linear','2022_05_18__16_58_59'),
                        ('nonlinear_johnson','2022_05_18__16_58_59'),
                        ('nonlinear_ours','2022_05_18__16_58_59')]

    eval(exp_date, exp_name, q_versions_and_dates, run=False)
