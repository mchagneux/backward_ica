import subprocess
import os
from datetime import datetime 
p_model = 'chaotic_rnn'
base_dir = os.path.join('experiments', f'p_{p_model}')


settings_list = ['johnson_backward,200.5.adam,1e-2,cst.reset,500,1.autodiff_on_backward.cpu.basic_logging']
num_fits = 1
num_epochs = 1000
dims = '5 5'
load_from_base_dir = 'data/crnn/dim_5_500_obs'
load_from_list = os.listdir(load_from_base_dir)

loaded_seq = True
batch_size = 1
num_seqs = 1
seq_length = 500
store_every = 0
 

for load_from in [os.path.join(load_from_base_dir, path) for path in load_from_list]:

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

         
    for process in processes: 
        process.wait()

