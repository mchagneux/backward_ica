import subprocess
import os
from datetime import datetime 

p_model = 'chaotic_rnn'
base_dir = os.path.join('experiments', f'p_{p_model}')

q_models = ['johnson_backward', 'johnson_forward']

num_epochs = 1000
learning_rate = 0.01
dims = '5 5'
load_from = '../online_var_fil/outputs/2022-10-25_16-50-34_Train_run'
batch_size = 10
num_seqs = 1000
seq_length = 50
num_samples = 1

os.makedirs(base_dir, exist_ok=True)


date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

exp_dir = os.path.join(base_dir, date)
os.makedirs(exp_dir, exist_ok=True)
subprocess.run(f'python generate_data.py \
                        --model {p_model} \
                        --dims {dims} \
                        --num_seqs {num_seqs} \
                        --seq_length {seq_length} \
                        --load_from {load_from} \
                        --exp_dir {exp_dir}',  
                shell=True)


processes = [subprocess.Popen(f'python train.py \
                                    --model {model} \
                                    --exp_dir {exp_dir} \
                                    --batch_size {batch_size} \
                                    --learning_rate {learning_rate} \
                                    --num_epochs {num_epochs} \
                                    --num_samples {num_samples}', 
                        shell=True) for model in q_models]

         
tensorboard_process = subprocess.Popen(f'tensorboard --logdir {exp_dir}', shell=True)
tensorboard_process.wait()

for process in processes: 
    process.wait()
