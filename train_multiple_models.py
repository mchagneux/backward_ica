import subprocess
import os
from datetime import datetime 

p_model = 'linear'
base_dir = os.path.join('experiments', f'p_{p_model}')

q_models = ['linear_online', 'linear_mc']

num_epochs = 250
learning_rate = 0.01
dims = '5 5'
load_from = ''
batch_size = 100
num_seqs = 100
seq_length = 50
num_samples_list = [10,10]
loaded_seq = False
sweep_sequences = False
store_every = 1
online_list = [True, False]
os.makedirs(base_dir, exist_ok=True)


date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

loaded_seq = '--loaded_seq' if loaded_seq else ''
load_from = f'--load_from {load_from}' if load_from != '' else ''
sweep_sequences = f'--sweep_sequences' if sweep_sequences else ''
online_list = [f'--online' if online else '' for online in online_list]

exp_dir = os.path.join(base_dir, date)
os.makedirs(exp_dir, exist_ok=True)

subprocess.run(f'python generate_data.py {loaded_seq} {load_from} \
                        --model {p_model} \
                        --dims {dims} \
                        --num_seqs {num_seqs} \
                        --seq_length {seq_length} \
                        --exp_dir {exp_dir}',  
                shell=True)


processes = [subprocess.Popen(f'python train.py {sweep_sequences} {online} \
                                --model {model} \
                                --exp_dir {exp_dir} \
                                --batch_size {batch_size} \
                                --learning_rate {learning_rate} \
                                --num_epochs {num_epochs} \
                                --store_every {store_every} \
                                --num_samples {num_samples}', 
                            shell=True) for model, num_samples, online in zip(q_models, 
                                                                        num_samples_list, 
                                                                        online_list)]

         
tensorboard_process = subprocess.Popen(f'tensorboard --logdir {exp_dir}', shell=True)
tensorboard_process.wait()

for process in processes: 
    process.wait()
