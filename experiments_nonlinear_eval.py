import argparse
import jax 
from backward_ica import hmm 
from backward_ica import utils
import matplotlib.pyplot as plt 
import pickle
utils.enable_x64(True)


def run_ffbsi_em_on_train_data(train_args, save_dir):

    key_theta = jax.random.PRNGKey(train_args.seed_theta)
    hmm.HMM.parametrization = train_args.parametrization
    utils.GaussianParams.parametrization = train_args.parametrization 
    key_params, key_gen, key_mle = jax.random.split(key_theta, 3)

    p = hmm.NonLinearGaussianHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                            layers=train_args.emission_map_layers,
                            slope=train_args.slope,
                            num_particles=train_args.num_particles) # specify the structure of the true model
    
    theta_star = p.get_random_params(key_params)
    utils.save_params(theta_star, 'theta', save_dir)

    obs_seqs = p.sample_multiple_sequences(key_gen, theta_star, train_args.num_seqs, train_args.seq_length)[1]
    
    print('Starting FFBSi EM training...')
    theta_mle, logls_mle = p.fit_ffbsi_em(key_mle, 
                                        obs_seqs, 
                                        optimizer=train_args.optimizer, 
                                        learning_rate=train_args.learning_rate, 
                                        batch_size=train_args.batch_size, 
                                        num_epochs=train_args.num_epochs)

    utils.save_params(theta_mle, 'phi', save_dir)

    utils.save_train_logs((0, None, [logls_mle]), save_dir, plot=False)

def run_ffbsi_smoothing_on_eval_data(train_args, eval_args):


    key = jax.random.PRNGKey(eval_args.seed)

    p = hmm.NonLinearGaussianHMM(state_dim=train_args.state_dim, 
                            obs_dim=train_args.obs_dim, 
                            transition_matrix_conditionning=train_args.transition_matrix_conditionning,
                            layers=train_args.emission_map_layers,
                            slope=train_args.slope,
                            num_particles=eval_args.num_particles) # specify the structure of the true model

    theta = utils.load_params('theta', train_args.save_dir)
    key_gen, key_ffbsi = jax.random.split(key,2)
    obs_seqs = p.sample_multiple_sequences(key_gen, theta, eval_args.num_seqs, eval_args.seq_length)[1]
    timesteps = range(1, eval_args.seq_length, eval_args.step)
    theta_mle = utils.load_params('phi', train_args.save_dir)

    smoothing_ffbsi = utils.multiple_length_ffbsi_smoothing(key_ffbsi, obs_seqs, p, theta_mle, timesteps)

    with open(os.path.join(eval_args.save_dir, 'smoothing_results'), 'wb') as f:
        pickle.dump(smoothing_ffbsi, f)

def compare(train_dirs, eval_dirs, method_names, save_dir):

    
    train_logs = [utils.load_train_logs(train_dir) for train_dir in train_dirs]
    # eval_args_list = [utils.load_args('eval_args', eval_dir) for eval_dir in eval_dirs]
    
    avg_evidence = train_logs[0][-1]
    avg_elbos_list = [train_log[2][train_log[0]] for train_log in train_logs]

    utils.superpose_training_curves(avg_evidence, avg_elbos_list, method_names, save_dir, start_index=0)

    # print(logls_mle[-1])
    # plt.plot(logls_mle)
    # plt.savefig(os.path.join(save_dir, 'loss_mle'))
    # plt.clf()

    

    

    # timesteps = range(1, eval_args.seq_length, eval_args.step)




if __name__ == '__main__': 

    import subprocess
    import os
    from datetime import datetime 
    exp_detail = 'p_noninjective'
    base_dir = os.path.join('experiments', 'p_nonlinear', exp_detail)

    q_versions_and_dates = [('linear','2022_05_14__12_56_50'),
                        ('nonlinear_johnson','2022_05_14__12_56_50'),
                        ('nonlinear_ours','2022_05_14__12_56_50')]

    train_dirs = []
    eval_dirs = []
    for (version, date) in q_versions_and_dates:
        train_dir = os.path.join(base_dir, 'trainings', version, date)
        eval_dir = os.path.join(base_dir, 'evals', f'{version}_{date}')
        os.makedirs(eval_dir, exist_ok=True)
        train_dirs.append(train_dir)
        eval_dirs.append(eval_dir)

    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')


    train_args = utils.load_args('train_args', train_dirs[0])
    mle_name = 'ffbsi_em'
    mle_save_dir = os.path.join(base_dir, 'trainings', mle_name, date)
    mle_train_args = argparse.Namespace()
    mle_train_args.num_particles = 2
    mle_train_args.learning_rate = 1e-1
    mle_train_args.batch_size = train_args.num_seqs
    mle_train_args.q_version = 'mle'

    mle_train_args.state_dim = train_args.state_dim 
    mle_train_args.obs_dim = train_args.obs_dim 
    mle_train_args.emission_map_layers = train_args.emission_map_layers 
    mle_train_args.slope = train_args.slope 
    mle_train_args.transition_matrix_conditionning =  train_args.transition_matrix_conditionning 
    mle_train_args.seed_theta = train_args.seed_theta 
    mle_train_args.parametrization = train_args.parametrization
    mle_train_args.num_seqs = train_args.num_seqs 
    mle_train_args.seq_length = train_args.seq_length
    mle_train_args.optimizer = train_args.optimizer
    mle_train_args.num_epochs = train_args.num_epochs
    mle_train_args.save_dir = mle_save_dir




    
    processes = [subprocess.Popen(f'python eval_nonlinear.py \
                                --train_dir {train_dir} \
                                --save_dir {eval_dir}', 
                            shell=True) \
                    for (train_dir, eval_dir) in zip(train_dirs, eval_dirs)]

    for process in processes:
        process.wait()

    os.makedirs(mle_save_dir)

    utils.save_args(mle_train_args, 'train_args', mle_save_dir)   

    run_ffbsi_em_on_train_data(mle_train_args, mle_save_dir)

    ffbsi_smoothing_save_dir = os.path.join(base_dir, 'evals', f'{mle_name}_{date}')
    os.makedirs(ffbsi_smoothing_save_dir)
    eval_args = utils.load_args('eval_args', eval_dirs[0])
    eval_args_ffbsi_smoothing = argparse.Namespace()
    eval_args_ffbsi_smoothing.save_dir = ffbsi_smoothing_save_dir
    eval_args_ffbsi_smoothing.num_seqs = eval_args.num_seqs 
    eval_args_ffbsi_smoothing.seq_length = eval_args.seq_length
    eval_args_ffbsi_smoothing.seed = eval_args.seed
    eval_args_ffbsi_smoothing.step = eval_args.step
    eval_args_ffbsi_smoothing.num_particles = mle_train_args.num_particles

    utils.save_args(eval_args_ffbsi_smoothing, 'eval_args', ffbsi_smoothing_save_dir)
    run_ffbsi_smoothing_on_eval_data(mle_train_args, eval_args_ffbsi_smoothing)



    train_dirs += [mle_save_dir]
    method_names = [q_version_and_date[0] for q_version_and_date in q_versions_and_dates] + [f'{mle_name}_{date}']
    

    save_dir = os.path.join(base_dir, 'evals', f'comparison_{date}')
    os.makedirs(save_dir)
    with open(os.path.join(save_dir, 'method_names.txt'), 'w') as f: 
        f.write(str([eval_dir.split('/')[-1] for eval_dir in eval_dirs]))

    compare(train_dirs, eval_dirs, method_names, save_dir)





