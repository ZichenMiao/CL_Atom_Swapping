import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from torch.nn.parameter import Parameter
import math
import scipy as sp
import scipy.linalg as linalg
import numpy as np
import pdb
from torch.nn.utils import spectral_norm




class Conv_DCFDE(nn.Module):
    """
      Modify to CL version: take coefficient during feed forward
    """

    __constants__ = ['kernel_size', 'stride', 'padding', 'num_bases',
                     'bases_grad', 'mode']
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                    num_bases=6, bases_grad=True, dilation=1, groups=1, mode='mode1', 
                    bases_drop=None, num_member=4):
        super(Conv_DCFDE, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # self.edge = (kernel_size-1)/2
        self.stride = stride
        self.padding = padding
        self.num_bases = num_bases
        assert mode in ['mode0', 'mode1'], 'Only mode0 and mode1 are available at this moment.'
        self.mode = mode
        self.bases_grad = bases_grad
        self.dilation = dilation
        self.bases_drop = bases_drop
        self.groups = groups
        self.num_member = num_member

        """
          Each layer only has bases/ bases' variational params
        """
        self.bases = Parameter(torch.Tensor(num_member, num_bases, kernel_size, kernel_size).normal_(0.0, 0.01))


    def forward(self, input):
        """
          For Bayesian Atoms Enseblme: Assign bases_mu & bases_logsig before forward pass.
          For Atoms Ensemble: Assign bases 
        """
        N, C, H, W = input.shape
        H = H//self.stride
        W = W//self.stride

        M = self.num_bases
        K = self.kernel_size

        E = self.num_member
        N = N//E
        stride = self.stride
        if self.kernel_size != 1:
                bases = self.bases
                coef = self.coef.view(self.out_channels, self.in_channels, self.num_bases)
                ## filters with shape of [num_member*chn_out, chn_in, k, k]
                filters = torch.einsum('cvm, emki-> ecvki', coef, bases).reshape(E*self.out_channels, -1, K, K)


        if self.kernel_size != 1:
            try:
                ## input [N//num_member, num_member*chn_in, H, W]
                x = input.view(E, N, *input.shape[1:]).permute(1,0,2,3,4).reshape(N, E*self.in_channels, *input.shape[2:])
            except:
                pdb.set_trace()
            # pdb.set_trace()
            ## filters with shape [num_member*chn_out, chn_in, k, k]
            ## out with shape [N//num_member, num_member*chn_out, H', W']
            out = F.conv2d(x, filters.contiguous(), stride=self.stride, padding=self.padding, groups=self.num_member)
            ## reshape to [N, chn_out, H', W']
            out = out.view(-1, E, self.out_channels, H, W).permute(1,0,2,3,4).reshape(-1, self.out_channels, H, W)
            
        else:
            out = F.conv2d(input, self.coef, None, stride=stride)

        return out


    def extra_repr(self):
        return 'kernel_size={kernel_size}, num_member={num_member}, stride={stride}, padding={padding}, num_bases={num_bases}' \
            ', bases_grad={bases_grad}, mode={mode}, in_channels={in_channels}, out_channels={out_channels}'.format(**self.__dict__)



if __name__ == '__main__':
    layer = Conv_DCFDE(1, 3, kernel_size=3, padding=1, stride=2).cuda()
    # layer = nn.Conv2d(3, 10, kernel_size=3, padding=1, stride=2).cuda()
    data = torch.randn(1 , 1, 4, 4).cuda()
    print(layer(data))
    print(layer(data))


