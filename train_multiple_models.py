import subprocess
import os
from datetime import datetime 

p_version = 'chaotic_rnn'
base_dir = os.path.join('experiments', f'p_{p_version}')

q_versions = ['neural_backward_linear']

num_epochs = 10000
learning_rate = 0.01
dims = '5 5'
load_sequences = True
sweep_sequences = False

batch_size = 1
num_seqs = 1
seq_length = 2000
num_samples = 1
float64 = False

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

load_sequences = '--load_sequences' if load_sequences else ''
sweep_sequences ='--sweep_sequences' if sweep_sequences else ''
float64 = '--float64' if float64 else ''
processes = [subprocess.Popen(f'python train.py {load_sequences} {sweep_sequences} {float64} \
                                --p_version {p_version} \
                                --q_version {q_version} \
                                --save_dir {save_dir} \
                                --learning_rate {learning_rate} \
                                --num_epochs {num_epochs} \
                                --seq_length {seq_length} \
                                --num_seqs {num_seqs} \
                                --batch_size {batch_size} \
                                --num_samples {num_samples} \
                                --dims {dims}',
                        shell=True, stdout=logfile, stderr=logfile) \

            for (q_version, 
                save_dir, 
                logfile) in zip(q_versions, 
                                save_dirs, 
                                logfiles)]

# print(date)

tensorboard_process = subprocess.Popen(f'tensorboard --logdir {base_dir}/{date}', shell=True)
tensorboard_process.wait()

for process in processes: 
    process.wait()



for logfile in logfiles:
    logfile.close()

