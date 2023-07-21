import subprocess
import os
from datetime import datetime 
p_model = 'chaotic_rnn'
base_dir = os.path.join('experiments', f'p_{p_model}')


settings_list = ['neural_backward_explicit_transition,8_8.5.adam,1e-2,cst.reset,500,1.autodiff_on_backward.cpu.basic_logging']
num_fits = 1
num_epochs = 1000
dims = '5 5'
load_from = 'data/crnn/2022-10-18_15-28-00_Train_run'
loaded_seq = True
num_seqs = 1
seq_length = 500
store_every = 0
 
os.makedirs(base_dir, exist_ok=True)

date = datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

loaded_seq = '--loaded_seq' if loaded_seq else ''
load_from = f'--load_from {load_from}' if load_from != '' else ''

exp_dir = os.path.join(base_dir, date)

subprocess.run(f'python generate_toy_sequential_data.py {loaded_seq} {load_from} \
                        --model {p_model} \
                        --dims {dims} \
                        --num_seqs {num_seqs} \
                        --seq_length {seq_length} \
                        --exp_dir {exp_dir}',  
                shell=True)




processes = [subprocess.Popen(f'python train_sequential_model.py \
                                --settings {settings} \
                                --exp_dir {exp_dir} \
                                --num_fits {num_fits} \
                                --num_epochs {num_epochs} \
                                --subseq_length {seq_length} \
                                --store_every {store_every}',
                            shell=True) for settings in settings_list]

tensorboard_process = subprocess.Popen(f'tensorboard --logdir {exp_dir}', shell=True)
tensorboard_process.wait()


for process in processes: 
    process.wait()


