import torch
import torch.nn as nn
import numpy as np
from functools import partial

from .Conv_DCFE import *
from .ensemble_layers import *


class Net(nn.Module):
    def __init__(self, num_classes=10, num_bases=12, num_member=1):
        super(Net, self).__init__()
        self.planes_list = [3, 64, 192, 384, 256, 256]
        self.pool_list = [0, 1, 4]
        self.hidden1 = 128

        ## dcf args
        self.num_bases = num_bases
        self.num_member = num_member

        
        self.maxpool=torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu=torch.nn.ReLU()

        self.drop2=torch.nn.Dropout(0.5)
        self.out_size = self.planes_list[-1] * (int(32 / 2**len(self.pool_list))**2)


        self.branch_list = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.added_bks = []

        self.Conv_Layer = partial(Conv_DCFDE, num_member=num_member, num_bases=num_bases,
                                bases_drop=False, bnn=False)

        """
          Create a single set of coefficient, with length=31 for resnet32
        """
        self.coeff_list = nn.ParameterList()
        ## first convolution coefficient if use ensemble_dcf_conv

        for i in range(len(self.planes_list)-1):
            self.coeff_list.append(Parameter(
                torch.Tensor(self.planes_list[i+1], self.planes_list[i]*num_bases, 1, 1)))
    
        ## initialize
        for pm in self.coeff_list:
            nn.init.kaiming_normal_(pm)


    def add_branch(self, num_outputs, block_expand=0):
        """
          add a set of new modules that form another branch
        """
        ## new fc
        self.heads.append(nn.Sequential(
            EsmLinear(self.out_size, num_outputs, num_member=self.num_member),
            # nn.ReLU(inplace=True),
            # nn.Dropout(),
            # EsmLinear(self.hidden1, num_outputs, num_member=self.num_member)
        )) 
        
        ## new convs 
        conv_blocks = nn.ModuleList()
        for i in range(len(self.planes_list)-1):
            conv_blocks.append(self.Conv_Layer(self.planes_list[i], self.planes_list[i+1],
                                            kernel_size=3, padding=1))
        
        self.reset_params(conv_blocks)
        self.branch_list.append(conv_blocks)

    def reset_params(self, branch):
        for m in branch.modules():
            if isinstance(m, Conv_DCFDE):
                nn.init.kaiming_normal_(m.bases)
    
    def assign_coeff(self, task_id):
        branch_ = self.branch_list[task_id]
        cnt = 0
        for m in branch_.modules():
            if isinstance(m, Conv_DCFDE):
                m.coef = self.coeff_list[cnt]
                cnt += 1

    # def single_branch_forward(self, x, task_id):
    #     ## assign coefficient first
    #     # pdb.set_trace()
    #     self.assign_coeff(task_id)
    #     # pdb.set_trace()
    #     ## forward pass
    #     branch_ = self.branch_list[task_id]
    #     head_ = self.heads[task_id]

    #     for l, module in enumerate(branch_):
    #         if l >= self.pool:
    #             x = self.drop1(self.relu(module(x)))
    #         else:
    #             x = self.maxpool(self.drop1(self.relu(module(x))))

    #     x = self.gap(x)
    #     x = x.view(x.size(0), -1)
    #     out = head_(x)

    #     return out, x

    def single_branch_forward(self, x, task_id):
        ## assign coefficient first
        # pdb.set_trace()
        self.assign_coeff(task_id)
        # pdb.set_trace()
        ## forward pass
        branch_ = self.branch_list[task_id]
        head_ = self.heads[task_id]

        for l, module in enumerate(branch_):
            x = self.relu(module(x))
            if l in self.pool_list:
                x = self.maxpool(x)

        # pdb.set_trace()
        x = x.view(x.size(0), -1)
        out = head_(x)

        return out, x


    def forward(self, x, task_id=None):
        # pdb.set_trace()
        if task_id is not None:
            """
              During training, current task id is provided
            """
            out, feat = self.single_branch_forward(x, task_id)

        else:
            """
              During testing, no task id is provided.
              We adopt an simple-minded solution temporarily.
            """
            out = []
            feat = []
            for task_ in range(len(self.heads)):
                out_, feat_ = self.single_branch_forward(x, task_)
                out.append(out_)
                feat.append(feat_)
            ## with shape [num_current_tasks, bs, num_cls]
            out = torch.stack(out, dim=0)
            feat = torch.stack(feat, dim=0)

        ## dimensions of outputs
        ## out: if task_id is not provided, with shape [bs, cls], else [num_tasks, bs, cls]
        ## feat: .....                    , with shape [bs, out_dim], else [num_tasks, bs, out_dim]
        
        return out, feat