import subprocess
import os
from datetime import datetime 
injective = True
exp_detail = 'p_nonlinear_dim_2_2'
base_dir = os.path.join('experiments', 'p_nonlinear', exp_detail, 'trainings')

q_versions = ['nonlinear_johnson', 'nonlinear_general_explicit_backward']

os.makedirs(base_dir, exist_ok=True)

logfiles = []
save_dirs = []
for q_version in q_versions:
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    save_dir = os.path.join(base_dir, q_version, date)
    os.makedirs(save_dir, exist_ok=True)
    f = open(os.path.join(save_dir, 'logfile.txt'),'w')

    logfiles.append(f)
    save_dirs.append(save_dir)


arg = '--injective' if injective else ''

processes = [subprocess.Popen(f'python train_nonlinear.py \
                            --q_version {q_version} \
                            --save_dir {save_dir} {arg}',
                        shell=True, stdout=logfile, stderr=logfile) \
                for (q_version, save_dir, logfile) in zip(q_versions, save_dirs, logfiles)]


tensorboard_process = subprocess.Popen(f'tensorboard --logdir {base_dir}', shell=True)
tensorboard_process.wait()

for process in processes: 
    process.wait()



for logfile in logfiles:
    logfile.close()

