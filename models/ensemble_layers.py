import torch
from torch import nn 
import torch.nn.functional as F
import pdb
# from torch.nn.parameter import Parameter
from torch.nn import Module, Parameter
import torch.nn.init as init
import math


class EsmBatchNorm2d(nn.Module):
    def __init__(self, num_channles, num_member=1):
        super(EsmBatchNorm2d, self).__init__()
        # self.bns = nn.ModuleList([nn.BatchNorm2d(num_channles) for _ in range(num_member)])
        for i in range(num_member):
            setattr(self, 'bn%d'%i, nn.BatchNorm2d(num_channles))
        self.num_member = num_member
        self.mem_idx = None

    def forward(self, x):
        if self.mem_idx is None:
            Bs = x.shape[0]

            N = Bs // self.num_member
            x = x.split(N, dim=0)

            out = []
            for i in range(self.num_member):
                # out.append(self.bns[i](x[i]))
                out.append(getattr(self, 'bn%d'%i)(x[i]))

            return torch.cat(out, dim=0)
        else:
            print('Using BN %d' %self.mem_idx)
            return getattr(self, 'bn%d' %self.mem_idx)(x)


class EsmLinear(nn.Module):
    def __init__(self, in_channles, out_channles, num_member=1):
        super(EsmLinear, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(in_channles, out_channles) for _ in range(num_member)])
        self.num_member = num_member
        self.out_channles = out_channles
        self.mem_idx = None

    def forward(self, x):
        if self.mem_idx is None:
            Bs = x.shape[0]

            N = Bs // self.num_member
            x = x.split(N, dim=0)

            out = []
            for i in range(self.num_member):
                out.append(self.linears[i](x[i]))

            return torch.cat(out, dim=0)

        else:
            return self.linears[self.mem_idx](x)
