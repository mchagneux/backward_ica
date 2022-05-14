import subprocess
import os
from datetime import datetime 
exp_detail = 'p_noninjective'
base_dir = os.path.join('experiments', 'p_nonlinear', exp_detail, 'trainings')

q_versions = ['linear',
            'nonlinear_johnson',
            'nonlinear_ours']

os.makedirs(base_dir)

logfiles = []
save_dirs = []

for q_version in q_versions:
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    save_dir = os.path.join(base_dir, q_version, date)
    os.makedirs(save_dir)
    f = open(os.path.join(save_dir, 'stdout.txt'),'w')

    logfiles.append(f)
    save_dirs.append(save_dir)

processes = [subprocess.Popen(f'python train_nonlinear.py \
                            --q_version {q_version} \
                            --save_dir {save_dir}', 
                        shell=True, stdout=logfile, stderr=logfile) \
                for (q_version, save_dir, logfile) in zip(q_versions, save_dirs, logfiles)]

for process in processes: 
    process.wait()

for logfile in logfiles:
    logfile.close()

