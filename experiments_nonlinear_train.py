import subprocess
import os
from datetime import datetime 

p_version = 'chaotic_rnn'
base_dir = os.path.join('experiments', f'p_{p_version}')

q_versions = ['johnson_freeze__covariances__prior_phi']

learning_rates = ['0.0001']
num_epochs_list = ['30000']
dims_list = ['5 5']

os.makedirs(base_dir, exist_ok=True)

logfiles = []
save_dirs = []
for q_version in q_versions:
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    save_dir = os.path.join(base_dir, date, q_version)
    os.makedirs(save_dir, exist_ok=True)
    f = open(os.path.join(save_dir, 'logfile.txt'),'w')

    logfiles.append(f)
    save_dirs.append(save_dir)


processes = [subprocess.Popen(f'python train_nonlinear.py \
                            --p_version {p_version} \
                            --q_version {q_version} \
                            --save_dir {save_dir} \
                            --learning_rate {learning_rate} \
                            --num_epochs {num_epochs} \
                            --dims {dims}',
                        shell=True, stdout=logfile, stderr=logfile) \
                        for (q_version, 
                            learning_rate, 
                            dims, 
                            num_epochs, 
                            save_dir, 
                            logfile) in zip(q_versions, 
                                            learning_rates, 
                                            dims_list, 
                                            num_epochs_list, 
                                            save_dirs, 
                                            logfiles)]

# print(date)

tensorboard_process = subprocess.Popen(f'tensorboard --logdir {base_dir}/{date}', shell=True)
tensorboard_process.wait()

for process in processes: 
    process.wait()



for logfile in logfiles:
    logfile.close()

