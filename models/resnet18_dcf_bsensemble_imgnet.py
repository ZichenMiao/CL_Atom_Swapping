import torch
import torch.nn as nn
import numpy as np
from functools import partial

from .Conv_DCFE import *
from .ensemble_layers import *


class First_Conv(nn.Module):
    """
      First convolution module in ResNet
    """
    def __init__(self, inplanes=3, planes=16, stride=1, num_bases=12, num_member=1):
        super().__init__()

        self.conv_module = nn.Sequential(
                                nn.Conv2d(inplanes, planes, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(planes),
                                nn.ReLU(inplace=True),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                            )

    def forward(self, x):
        return self.conv_module(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, num_bases=12, num_member=1):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        conv_layer = partial(Conv_DCFDE, num_member=num_member, num_bases=num_bases)
        bn = partial(EsmBatchNorm2d, num_member=num_member)

        self.conv1 = conv_layer(inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_layer(planes, planes, kernel_size=3, padding=1)
        self.bn2 = bn(planes)
        self.stride = stride

        self.downsample = None
        if stride != 1 or self.inplanes != self.planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                bn(planes * self.expansion),
            )

    def forward(self, x):
        ## bases with shape [2, k, k, num_bases]
        residual = x
        # pdb.set_trace()
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, num_bases=12, num_member=1, parallel=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.block = block
        self.layers = num_blocks
        self.stride_list = [1, 2, 2, 2]
        self.planes_list = [64, 128, 256, 512]
        self.parallel = parallel

        ## dcf args
        self.num_bases = num_bases
        self.num_member = num_member

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.out_size = 512 * block.expansion

        self.branch_list = nn.ModuleList()
        self.heads = nn.ModuleList()
        self.added_bks = []


        """
          Create a single set of coefficient, with length=31 for resnet32
        """
        coeff_list = nn.ParameterList()
        inplanes = self.in_planes
        for l, planes in enumerate(self.planes_list):
            for b in range(self.layers[l]):
                ## conv1 coeff
                coeff_list.append(Parameter(torch.Tensor(planes, inplanes*num_bases, 1, 1)))
                ## update inplanes
                if b == 0:
                    inplanes = planes * self.block.expansion
                
                ## conv2 coeff
                coeff_list.append(Parameter(torch.Tensor(planes, planes*num_bases, 1, 1)))

        ## initialize
        for pm in coeff_list:
            nn.init.kaiming_normal_(pm)

        ## parallel
        if self.parallel:
            for i, param in enumerate(coeff_list):
                self.register_parameter(f'coeff_list_{i}', param)
        else:
            self.coeff_list = coeff_list
        

    def add_branch(self, num_outputs):
        """
          add a set of new modules that form another branch
        """
        ## new fc
        self.heads.append(EsmLinear(self.out_size, num_outputs, num_member=self.num_member))
        self.task_cls = torch.tensor([10 for head in self.heads])
        
        ## all other layers
        conv_blocks = nn.ModuleList()
        #  first conv module
        conv_blocks.append(First_Conv(inplanes=3, planes=self.in_planes, num_bases=self.num_bases, 
                                        num_member=self.num_member))
        #  4 layers of blocks
        #  initial inplanes
        inplanes = self.in_planes
        for i in range(4):
            stride_ = self.stride_list[i]
            planes = self.planes_list[i]
            num_blocks = self.layers[i]
            ## add the first block with potential downsample
            conv_blocks.append(self.block(inplanes, planes, stride=stride_, 
                                            num_bases=self.num_bases, num_member=self.num_member))
            inplanes = planes * self.block.expansion
            for l in range(1, num_blocks):
                ## add the rest of blocks
                conv_blocks.append(self.block(inplanes, planes, num_bases=self.num_bases,
                                            num_member=self.num_member))
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
                if self.parallel:
                    m.coef = getattr(self, f'coeff_list_{cnt}')
                else:
                    m.coef = self.coeff_list[cnt]
                cnt += 1

    def single_branch_forward(self, x, task_id):
        ## assign coefficient first
        self.assign_coeff(task_id)
        ## forward pass
        branch_ = self.branch_list[task_id]
        head_ = self.heads[task_id]

        for l, module in enumerate(branch_):
            x = module(x)

        x = self.avgpool(x)
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
            # pdb.set_trace()
            out = []
            feat = []
            for task_ in range(len(self.heads)):    
                out_, feat_ = self.single_branch_forward(x, task_)
                out.append(out_)
                feat.append(feat_)
        
        return out, feat


def Net(**args):
    return ResNet(BasicBlock, [2,2,2,2], **args)
