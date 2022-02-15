import nntplib
import torch 
import torch.nn as nn
from torch.nn.utils.parametrize import register_parametrization


class Diag(nn.Module):
    def forward(self, X):
        return torch.diag(X) ** 2 # Return a symmetric matrix

    def right_inverse(self, A):
        return torch.sqrt(torch.diag(A))

class Det(nn.Module):

    def forward(self, X):
        return torch.det(X)

class Inv(nn.Module):
    def forward(self, X):
        return torch.inverse(X)

    def right_inverse(self, A):
        return torch.inverse(A)


cov1 = nn.parameter.Parameter(torch.rand(2))

cov_dict = nn.ParameterDict({'cov1':cov1})

parameter_dict = nn.ModuleDict({'covs':cov_dict})

register_parametrization(parameter_dict.covs, 'cov1', Diag())



otherModule = nn.ModuleList([parameter_dict])
parameter_dict.covs.cov1 = torch.tensor([[0.01,0.2],[0,0.01]])
print('Initialized at:',parameter_dict.covs.cov1)
parameter_dict.covs.cov1 -= torch.tensor([[0.005,0.2],[0,0.005]])
print('After update:',parameter_dict.covs.cov1)
for name, param in parameter_dict.named_parameters(): 
    if name == 'covs.parametrizations.cov1.original': print(param)


