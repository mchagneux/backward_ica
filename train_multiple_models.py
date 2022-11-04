import subprocess
import os
from datetime import datetime 

p_model = 'linear'
base_dir = os.path.join('experiments', f'p_{p_model}')

q_models = ['linear_online']

num_epochs = 200
learning_rate = 0.01
dims = '5 5'
load_from = ''
batch_size = 100
num_seqs = 1000
seq_length = 500
num_samples_list = [0]
loaded_seq = False
sweep_sequences = False
store_every = 5
os.makedirs(base_dir, exist_ok=True)


date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

loaded_seq = '--loaded_seq' if loaded_seq else ''
load_from = f'--load_from {load_from}' if load_from != '' else ''
sweep_sequences = f'--sweep_sequences' if sweep_sequences else ''

exp_dir = os.path.join(base_dir, date)
os.makedirs(exp_dir, exist_ok=True)

subprocess.run(f'python generate_data.py {loaded_seq} {load_from} \
                        --model {p_model} \
                        --dims {dims} \
                        --num_seqs {num_seqs} \
                        --seq_length {seq_length} \
                        --exp_dir {exp_dir}',  
                shell=True)


processes = [subprocess.Popen(f'python train.py {sweep_sequences} \
                                    --model {model} \
                                    --exp_dir {exp_dir} \
                                    --batch_size {batch_size} \
                                    --learning_rate {learning_rate} \
                                    --num_epochs {num_epochs} \
                                    --store_every {store_every} \
                                    --num_samples {num_samples}', 
                        shell=True) for model, num_samples in zip(q_models, num_samples_list)]

         
tensorboard_process = subprocess.Popen(f'tensorboard --logdir {exp_dir}', shell=True)
tensorboard_process.wait()

for process in processes: 
    process.wait()
