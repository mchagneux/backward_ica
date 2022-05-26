import pandas as pd 
import json 

with open('experiments/p_nonlinear/p_noninjective/evals/comparisons/2022_05_24__13_51_29/mse_values.json') as f: 
    values = json.load(f)

print(values)

reference_errors_means = [value[0] for value in values['reference_errors']]
reference_errors_vars = [value[1] for value in values['reference_errors']]
errors_epoch_59 = [value[0] for value in values['Variational 59']]
errors_epoch_79 = [value[0] for value in values['Variational 79']]
errors_epoch_99 = [value[0] for value in values['Variational 99']]
df = pd.DataFrame([reference_errors_means, reference_errors_vars, errors_epoch_59, errors_epoch_79, errors_epoch_99])
print(df.T.to_latex(float_format="%.2f" ))