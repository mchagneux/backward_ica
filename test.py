import torch 
import torch.nn as nn


state_dim = 2 
param1 = nn.parameter.Parameter(torch.rand(state_dim))


matrix = nn.parameter.Parameter(torch.diag(torch.rand(state_dim)))
offset = nn.parameter.Parameter(torch.rand(state_dim))

transition = nn.Linear(in_features=state_dim, out_features=state_dim, bias=True)

transition.weight = nn.parameter.Parameter(torch.diag(torch.rand(state_dim)))
transition.bias  = nn.parameter.Parameter(torch.rand(state_dim))
transition.cov = nn.parameter.Parameter(torch.rand((state_dim,state_dim)))

test = 0 
