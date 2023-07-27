import subprocess
import os
from datetime import datetime 
p_model = 'nonlinear_emission'
base_dir = os.path.join('experiments', f'p_{p_model}')
logging_type = 'basic_logging'

settings_list = [f'conjugate_forward,8_8.5.adam,1e-2,cst.reset,50,1.autodiff_on_backward.gpu.{logging_type}',
                 f'johnson_backward,8_8.5.adam,1e-2,cst.reset,50,1.autodiff_on_backward.gpu.{logging_type}',
                 f'neural_backward_explicit_transition,8_8.5.adam,1e-2,cst.reset,50,1.autodiff_on_backward.gpu.{logging_type}']
num_fits = 1
num_epochs = 1000
dims = '10 10'
load_from = '' #data/crnn/2022-10-18_15-28-00_Train_run'
loaded_seq = False
num_seqs = 100
seq_length = 50
store_every = 0
parallel_experiments = False
 
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





if logging_type == 'tensorboard':
    tensorboard_process = subprocess.Popen(f'tensorboard --logdir {exp_dir}', shell=True)
    tensorboard_process.wait()
if parallel_experiments: 
    processes = [subprocess.Popen(f'python train_sequential_model.py \
                                    --settings {settings} \
                                    --exp_dir {exp_dir} \
                                    --num_fits {num_fits} \
                                    --num_epochs {num_epochs} \
                                    --subseq_length {seq_length} \
                                    --store_every {store_every}',
                                shell=True) for settings in settings_list]
    for process in processes: 
        process.wait()


else: 
    for settings in settings_list:
        subprocess.run(f'python train_sequential_model.py \
                                    --settings {settings} \
                                    --exp_dir {exp_dir} \
                                    --num_fits {num_fits} \
                                    --num_epochs {num_epochs} \
                                    --subseq_length {seq_length} \
                                    --store_every {store_every}',
                                shell=True)
