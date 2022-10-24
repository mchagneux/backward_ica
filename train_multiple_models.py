import subprocess
import os
from datetime import datetime 

p_model = 'chaotic_rnn'
base_dir = os.path.join('experiments', f'p_{p_model}')

q_models = ['johnson_backward', 'johnson_forward', 'neural_backward_linear']

num_epochs = 10000
learning_rate = 0.01
dims = '5 5'
load_from = '../online_var_fil/outputs/2022-10-24_12-02-05_Train_run'

batch_size = 1
num_seqs = 1
seq_length = 500
num_samples = 1

os.makedirs(base_dir, exist_ok=True)


date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

exp_dir = os.path.join(base_dir, date)
os.makedirs(exp_dir, exist_ok=True)

generate_logfile = open(os.path.join(exp_dir, 'generate_log.txt'),'w')
subprocess.run(f'python generate_data.py \
                        --model {p_model} \
                        --dims {dims} \
                        --num_seqs {num_seqs} \
                        --seq_length {seq_length} \
                        --load_from {load_from} \
                        --exp_dir {exp_dir}',  
                shell=True, 
                stdout=generate_logfile, 
                stderr=generate_logfile)
generate_logfile.close()
logfiles = []

for model in q_models: 
    os.makedirs(os.path.join(exp_dir, model), exist_ok=True)
    logfiles.append(open(os.path.join(exp_dir, model, 'train_log.txt'), 'w'))


processes = [subprocess.Popen(f'python train.py \
                                    --model {model} \
                                    --exp_dir {exp_dir} \
                                    --batch_size {batch_size} \
                                    --learning_rate {learning_rate} \
                                    --num_epochs {num_epochs} \
                                    --num_samples {num_samples}', 
                        shell=True, 
                        stdout=logfile, 
                        stderr=logfile) for model, logfile in zip(q_models, logfiles)]

         
tensorboard_process = subprocess.Popen(f'tensorboard --logdir {base_dir}/{date}', shell=True)
tensorboard_process.wait()

for process in processes: 
    process.wait()


for logfile in logfiles:
    logfile.close()

