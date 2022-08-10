import subprocess
import os
from datetime import datetime 
injective = True
base_dir = os.path.join('experiments', 'p_nonlinear')

q_versions = ['linear_freeze__theta',
            'johnson_freeze__theta',
            'johnson_explicit_proposal_freeze__theta']

learning_rates = ['0.01',
                '0.001',
                '0.01']

dims_list = ['1 1',
            '1 1',
            '1 1']
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


arg = '--injective' if injective else ''

processes = [subprocess.Popen(f'python train_nonlinear.py \
                            --q_version {q_version} \
                            --save_dir {save_dir} {arg} \
                            --learning_rate {learning_rate} \
                            --dims {dims}',
                        shell=True, stdout=logfile, stderr=logfile) \
                for (q_version, learning_rate, dims, save_dir, logfile) in zip(q_versions, learning_rates, dims_list, save_dirs, logfiles)]

# print(date)

tensorboard_process = subprocess.Popen(f'tensorboard --logdir {base_dir}/{date}', shell=True)
tensorboard_process.wait()

for process in processes: 
    process.wait()



for logfile in logfiles:
    logfile.close()

