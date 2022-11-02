import subprocess 

n_slices = 50 
exp_dirs = ['experiments/p_chaotic_rnn/2022_10_25__23_26_45', 
            'experiments/p_chaotic_rnn/2022_10_24__17_25_51',
            'experiments/p_chaotic_rnn/2022_10_25__15_53_00',
            'experiments/p_chaotic_rnn/2022_10_25__16_25_49',
            'experiments/p_chaotic_rnn/2022_10_25__16_52_19']

processes = [subprocess.Popen(f'python eval.py --exp_dir {exp_dir} --n_slices {n_slices}', shell=True) for exp_dir in exp_dirs]

for p in processes: 
    p.wait()

