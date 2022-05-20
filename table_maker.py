import pandas as pd 
import json 

with open('experiments/p_nonlinear/p_noninjective_2/evals/comparisons/2022_05_19__16_24_08/mse_values.json') as f: 
    values = json.load(f)

print(values)

means_ffbsi = [value[0] for value in values['ffbsi']]
vars_ffbsi = [value[1] for value in values['ffbsi']]
errors_johnson = [value[0] for value in values['nonlinear_johnson_2022_05_18__16_58_59']]
errors_ours = [value[0] for value in values['nonlinear_ours_2022_05_18__16_58_59']]

df = pd.DataFrame([means_ffbsi, vars_ffbsi, errors_johnson, errors_ours])
print(df.T.to_latex(float_format="%.2f" ))