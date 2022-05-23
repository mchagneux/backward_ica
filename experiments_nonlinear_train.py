import subprocess
import os
from datetime import datetime 
injective = False
exp_detail = 'p_noninjective_2'
base_dir = os.path.join('experiments', 'p_nonlinear', exp_detail, 'trainings')

q_versions = ['nonlinear_johnson']
            #'nonlinear_ours']

os.makedirs(base_dir, exist_ok=True)

logfiles = []
save_dirs = []
args_paths = []
for q_version in q_versions:
    date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')
    save_dir = os.path.join(base_dir, q_version, date)
    os.makedirs(save_dir, exist_ok=True)
    f = open(os.path.join(save_dir, 'stdout.txt'),'w')

    logfiles.append(f)
    save_dirs.append(save_dir)
    args_paths.append('archived_experiments/submission_19_05/p_nonlinear/p_noninjective_2/trainings/nonlinear_johnson/2022_05_18__16_58_59')


arg = '--injective' if injective else ''

processes = [subprocess.Popen(f'python train_nonlinear.py \
                            --q_version {q_version} \
                            --save_dir {save_dir} {arg} \
                            --args_path {args_path}',
                        shell=True, stdout=logfile, stderr=logfile) \
                for (q_version, save_dir, args_path, logfile) in zip(q_versions, save_dirs, args_paths, logfiles)]

for process in processes: 
    process.wait()

for logfile in logfiles:
    logfile.close()

