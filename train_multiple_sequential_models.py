import subprocess
import os
from datetime import datetime 
p_model = 'chaotic_rnn'
base_dir = os.path.join('experiments', f'p_{p_model}')

q_models = ['johnson_backward__online_50_reset_score_variance_reduction_bptt_depth_2',
            'johnson_backward__online_50_reset_autodiff_on_backward']


num_samples_list = [10, 10]
learning_rates = [1e-2, 1e-2]
num_epochs = 1000
dims = '5 5'
load_from = 'data/crnn/2022-10-18_15-28-00_Train_run'
loaded_seq = True
batch_size = 1
num_seqs = 1
seq_length = 2000
store_every = 0
 

optimizer = 'sgd'

os.makedirs(base_dir, exist_ok=True)

date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

loaded_seq = '--loaded_seq' if loaded_seq else ''
load_from = f'--load_from {load_from}' if load_from != '' else ''

exp_dir = os.path.join(base_dir, date)
os.makedirs(exp_dir, exist_ok=True)

subprocess.run(f'python generate_toy_sequential_data.py {loaded_seq} {load_from} \
                        --model {p_model} \
                        --dims {dims} \
                        --num_seqs {num_seqs} \
                        --seq_length {seq_length} \
                        --exp_dir {exp_dir}',  
                shell=True)


processes = [subprocess.Popen(f'python train_sequential_model.py \
                                --model {model} \
                                --exp_dir {exp_dir} \
                                --batch_size {batch_size} \
                                --learning_rate {learning_rate} \
                                --num_epochs {num_epochs} \
                                --store_every {store_every} \
                                --num_samples {num_samples} \
                                --optimizer {optimizer}', 
                            shell=True) \
                for model, num_samples, learning_rate in zip(
                                        q_models, 
                                        num_samples_list, 
                                        learning_rates)]

         
tensorboard_process = subprocess.Popen(f'tensorboard --logdir {exp_dir}', shell=True)
tensorboard_process.wait()

for process in processes: 
    process.wait()
