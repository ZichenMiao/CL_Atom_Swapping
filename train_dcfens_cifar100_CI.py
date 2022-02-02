import os
import sys
import torch
import torch.nn as nn
import numpy as np
import argparse
import copy
import pdb
import torch.nn.functional as F
import re, random, collections
import pickle
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import argparse
from torch.optim.lr_scheduler import MultiStepLR
torch.set_printoptions(precision=5,sci_mode=False)

from utils import logmeanexp
import incremental_dataloader as data
from utils import *
from models.Conv_DCFE import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default="./Datasets/ImageNet/")
parser.add_argument('--num_class', default=100, type=int)
parser.add_argument('--num_task', default=6, type=int, choices=[6, 11])
parser.add_argument('--first_task_cls', default=10, type=int)
parser.add_argument('--dataset', default='imagenet100')

parser.add_argument('--train_batch', default=128, type=int)
parser.add_argument('--test_batch', default=500, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--random_classes', action='store_true')
parser.add_argument('--validation', type=float, default=0.0)
parser.add_argument('--overflow', action='store_true')
parser.add_argument('--model')

parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--lr_sub', type=float, default=0.01, help='used after the 1st task')
parser.add_argument('--lr_schedule', default='80-120', help='learning rate drop schedule')
parser.add_argument('--lr_schedule_sub', default='80-120', help='learning rate drop schedule, used after the 1st task')
parser.add_argument('--total_epoch', default=160, type=int)
parser.add_argument('--total_epoch_sub', default=160, type=int, help='used after the 1st task')
parser.add_argument('--wd', default=5e-3, help='weight decay', type=float)
parser.add_argument('--wd_sub', default=5e-3, help='weight decay, used after the 1st task', type=float)

parser.add_argument('--optim', default='sgd', choices=['sgd', 'adam', 'nestrov'], help='Optimizer')
parser.add_argument('--gpu', default='0')
parser.add_argument('--init_with_pre', action='store_true', 
                        help='initialize the current task model parameters with the previous one')
parser.add_argument('--start_from', default=0, type=int, help='start CL from which task')

parser.add_argument('--num_bases', default=12, type=int)
parser.add_argument('--num_member', default=1, type=int)
parser.add_argument('--add_description', default='')

args = parser.parse_args()

## task per class
args.class_per_task = int(args.num_class // args.num_task)

from models.resnet32_dcf_bsensemble import Net

log_path = 'checkpoints'
exp_name = f'{args.dataset}_{args.num_task}tasks_firstcls{args.first_task_cls}_member{args.num_member}_{args.model}_bases{args.num_bases}_wd{args.wd}_{args.optim}'
exp_name += f'_{args.add_description}' if args.add_description else ''
args.model_path = os.path.join(log_path, exp_name)
os.makedirs(args.model_path, exist_ok=True)
## create checkpoint dir and copy files
file_dir = os.path.join(args.model_path, 'files')
os.makedirs(args.model_path, exist_ok=True)
os.makedirs(file_dir, exist_ok=True)
os.system(f'cp -r models/ idatasets/ *.py *.sh {file_dir}')

os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
set_seed(521)


## logger, copy stdout to a file
from logger import FileOutputDuplicator
sys.stdout = FileOutputDuplicator(sys.stdout, os.path.join(args.model_path, 'log.txt'), 'w')

print('Args:')
print(args)
print()
# print('seed: ', seed)


def train(train_loader, epoch, task, model, total_epoch):
    ## model: currently trained model, task_model: past models
    print('\nTask: %d, Epoch: %d' % (task, epoch))
    model.branch_list[task].train()
    model.heads[task].train()

    global best_acc
    metric_loss = 0
    min_entro_losses = 0
    train_loss = 0
    correct = 0
    total = 0

    previous_cls = sum(class_increments[:task])
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        targets=targets-previous_cls # assume sequential split for random  split mapping should me changed
        optimizer.zero_grad()
        
        ## outputs with shape [bs*num_member, num_cls], same
        bs = inputs.shape[0]
        outputs, feat_current = model(torch.cat([inputs] * args.num_member, dim=0), task_id=task)
        outputs = outputs.split(bs)
        loss = torch.sum(torch.stack([criterion(outputs_, targets) for outputs_ in outputs], dim=0), dim=0)
        
        loss.backward()
        
        optimizer.step()

        train_loss += loss.item()

        _, predicted = outputs[0].max(1)
        # _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        acc = 100.*correct/total
    
    print("[Train: ], [%d/%d: ], [Accuracy: %.2f], [Loss: %f], [Lr: %f]" 
          %(epoch, total_epoch, acc, train_loss/batch_idx, optimizer.param_groups[0]['lr']))



def ensemble_outputs(pre_outputs, bs):
    """
        pre_outputs: with batch_size repeated to batch_size * ensemble_numbers
        bs:          real batch_size
    """
    ## a list of outputs with length [num_member], each with shape [bs, num_cls]
    outputs = pre_outputs.split(bs)
    ## with shape [bs, num_cls, num_member]
    outputs = torch.stack(outputs, dim=-1)
    outputs = F.log_softmax(outputs, dim=-2)
    ## with shape [bs, num_cls]
    log_outputs = logmeanexp(outputs, dim=-1)

    return log_outputs


def get_optimizer(model, task_id):
    """
        train all parameters, or train atoms+heads only
    """

    parameters_branch = dict((model.branch_list[task_id]).named_parameters())
    parameters_branch_head = dict((model.heads[task_id]).named_parameters())

    ## new feat params
    parameters = [v for k, v in parameters_branch.items() if not ('coef' in k)]
    train_keys = [k for k, v in parameters_branch.items() if not ('coef' in k)]
    ## head parameters
    parameters += [v for k, v in parameters_branch_head.items()]
    train_keys += [k for k, v in parameters_branch_head.items()]

    if task_id == 0:
        ## learn coefficient
        parameters = parameters + list(model.coeff_list)
        train_keys = train_keys + [f'coeff_list.{i}' for i in range(len(model.coeff_list))]

    print('***Optimized Parameters:')
    # pdb.set_trace()
    print(', '.join(train_keys))

    if task_id > 0:
        branch_param_count = np.sum([param.numel() for param in parameters]) / 1e6
        print('\nNumber of Params in a New Branch: {:.2f}M'.format(branch_param_count))
        print('Added Memory per task: {:.2f}MB\n'.format(branch_param_count*4))

    if args.optim == 'sgd':
        if task_id == 0:
            optimizer = optim.SGD(parameters, lr=args.lr, momentum=0.9, weight_decay=args.wd)
        else:
            optimizer = optim.SGD(parameters, lr=args.lr_sub, momentum=0.9, weight_decay=args.wd_sub)
    elif args.optim == 'adam':
        optimizer = torch.optim.Adam(parameters, weight_decay=args.wd, lr=args.lr)
    elif args.optim == 'nestrov':
        optimizer = torch.optim.SGD(parameters, lr=args.lr, momentum=0.9, nesterov=True)
    else:
        raise NotImplementedError('...')

    return optimizer

def check_task(task, inputs, model, total_task='all'):
    """
      Task id classification
    """
    joint_entropy_tasks=[]
    model.eval()
    bs = inputs.shape[0]
    with torch.no_grad():
        ## list of tensors, each with shape [bs*num_member, num_cls_taski]
        preoutputs, _ = model(torch.cat([inputs]*args.num_member, 0), task_id=None)

        ## calculate entropy of every branch(task)
        for i, preout_ in enumerate(preoutputs):
            ## with shape [bs, num_cls_taski]
            outputs = ensemble_outputs(preout_, bs)
            outputs = torch.exp(outputs)

            ## get entropy -\sum_y y * log(y), with shape [bs]
            joint_entropy = -torch.sum(outputs * torch.log(outputs+0.0001), dim=1)
            """
                normailzing term for entropy given number of classes
            """
            p = class_increments[i] // min(class_increments)
            if args.num_task == 11 and i == 0:
                p *= 4
            joint_entropy /= p
            joint_entropy_tasks.append(joint_entropy)
    
        ## with shape [bs, num_current_task]
        joint_entropy_tasks = torch.stack(joint_entropy_tasks)
        joint_entropy_tasks = joint_entropy_tasks.transpose(0, 1)

    """
      constrain to previous seen tasks
    """
    if not total_task == 'all':
        joint_entropy_tasks = joint_entropy_tasks[:, :total_task]

    ## mask to indicate the correct task prediction
    ctask = torch.argmin(joint_entropy_tasks, axis=1)==task
    correct = sum(ctask)
    
    return ctask, correct, joint_entropy


def test(test_loader, task, model):
    global best_acc
    model.eval()
    test_loss = 0
    correct_ti = 0
    correct_ci = 0
    total = 0
    cl_loss=0
    tcorrect=0
    previous_cls = sum(class_increments[:task])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets1 = inputs.cuda(), targets.cuda()
            targets = targets1-previous_cls
            bs = inputs.shape[0]
            if task>0:
                ## for task_id > 0, get
                correct_sample, Ncorrect, _ = check_task(task, inputs, model)
                tcorrect += Ncorrect

            
            ## inference forward
            if inputs.shape[0]!=0:
                ## with shape [bs*num_member, num_cls]
                outputs, _ = model(torch.cat([inputs]*args.num_member, dim=0), task_id=task)
                ## ensemble results, with shape [bs, cls_per_task]
                outputs = ensemble_outputs(outputs, bs)
                
                loss = F.nll_loss(outputs, targets)
                test_loss += loss.item()
                
                ## ti
                _, predicted = outputs.max(1)
                correct_ti += predicted.eq(targets).sum().item()

                ## ci
                if task > 0:
                    predicted = predicted[correct_sample]
                    targets = targets[correct_sample]
                    correct_ci += predicted.eq(targets).sum().item()
            
            ## true batch size
            total += targets1.size(0)
    
    acc_ti = 100. * correct_ti / total
    if task > 0:
        taskC = tcorrect.item()/total
        acc_ci = 100. * correct_ci / total
        print("[Test CI Acc.: %.2f], [TI Acc.: %.2f] [Loss: %f] [Correct: %f]" %(acc_ci, acc_ti, 
                            test_loss/batch_idx, taskC))
    else:
        taskC = 1.0 
        acc_ci = acc_ti
        print("[Test TI Acc.: %.2f] [Loss: %f] [Correct: %f]" %(acc_ti, test_loss/batch_idx, taskC))
    
    ## model saving 
    if acc_ci >= best_acc:
        save_model(args, task, acc_ci, model)
        best_acc = acc_ci

    return acc_ci


def inferecne(test_loader, task, total_task, model):
    global best_acc
    model.eval()
    test_loss = 0
    correct_ti = 0 ## task class cls. correct
    correct_ci = 0
    total = 0
    cl_loss=0
    tcorrect=0  ## task id classification correct
    accuracy=[]
    previous_cls = sum(class_increments[:task])
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets1 = inputs.cuda(), targets.cuda()
            bs = inputs.shape[0]
            # pdb.set_trace()
            targets = targets1 - previous_cls
                        
            ### task id cls. 
            correct_sample, Ncorrect, _ = check_task(task, inputs, model, total_task)
            tcorrect += Ncorrect

                
            if inputs.shape[0]!=0:
                ## single branch forward, with shape [bs*num_member, cls_per_task]
                outputs, _ = model(torch.cat([inputs] * args.num_member, dim=0), task_id=task)
                ## ensemble results, log of softmax scores, with shape [bs, cls_per_task]
                outputs = ensemble_outputs(outputs, bs)

                loss = F.nll_loss(outputs, targets)
                test_loss += loss.item()

                ## ti
                _, predicted = outputs.max(1)
                correct_ti += predicted.eq(targets).sum().item()

                """
                select the samples with correct task id prediction
                """
                predicted = predicted[correct_sample]
                targets = targets[correct_sample]
                correct_ci += predicted.eq(targets).sum().item()
            
            ## true batch size
            total += targets1.size(0)
    
    taskC = tcorrect.item()/total

    acc_ti = 100.*correct_ti/total
    acc_ci = 100.*correct_ci/total

    # print("[Test CI Acc.: %.2f], [TI Acc.: %.2f] [Loss: %f] [Correct: %f]" %(acc_ci, acc_ti, 
    #                     test_loss/batch_idx, taskC))
    
    return correct_ci, total, tcorrect




## Dataloaders
inc_dataset = data.IncrementalDataset(
                                dataset_name=args.dataset,
                                args = args,
                                random_order=args.random_classes,
                                shuffle=True,
                                seed=1,
                                batch_size=args.train_batch,
                                workers=args.workers,
                                validation_split=args.validation,
                                increment=args.class_per_task,
                                first_task_cls=args.first_task_cls,
                                num_tasks=args.num_task
                            )
task_data=[]
for i in range(args.num_task):
    task_info, train_loader, val_loader, test_loader = inc_dataset.new_task()

    task_data.append([train_loader, test_loader])

class_increments = inc_dataset.increments

## initialize network
net = Net(num_classes=args.class_per_task, num_bases=args.num_bases, num_member=args.num_member)
if args.start_from > 0:
    for pre_t in range(args.start_from):
        task_cls_ = class_increments[pre_t]
        net.add_branch(task_cls_)
        net.cuda()
    ## load previous model (mainly load the coefficient)
    load_model_resume(args, pre_t, net)

## Training
###############################################

ci_acc_list=[]

## Loss
criterion = nn.CrossEntropyLoss()

for task in range(args.start_from, args.num_task):

    ### My version of training/ testing a task
    best_acc = 0
    print('Training Task :---'+str(task))

    ## dataloaders
    train_loader, test_loader = task_data[task][0],task_data[task][1]
    ## add a new network branch
    net.add_branch(class_increments[task], task)
    net.cuda()

    #  init model with previous task's params
    if task > 0 and args.init_with_pre:
        copy_head = class_increments[task] == class_increments[task-1]
        load_past(args, task, net, copy_first=False, copy_head=False)

    ## get optimizer
    optimizer = get_optimizer(net, task)
    if task == 0:
        schedule = args.lr_schedule
        schedule = [int(s) for s in schedule.split('-')]
        print('LR Drop Schedule: ', schedule)
        schedulerG = MultiStepLR(optimizer, milestones=schedule, gamma=0.1)
    else:
        schedule = args.lr_schedule_sub
        schedule = [int(s) for s in schedule.split('-')]
        print('LR Drop Schedule: ', schedule)
        schedulerG = MultiStepLR(optimizer, milestones=schedule, gamma=0.1)

    ## train-test 
    total_epoch = args.total_epoch if task == 0 else args.total_epoch_sub
    for epoch in range(total_epoch):
        train(train_loader, epoch, task, net, total_epoch)
        test(test_loader, task, net)
        schedulerG.step()

    ## restore the best model
    acc1 = load_model(args, task, net)
    # task_acc.append(acc1)
    # print('Task: '+str(task)+'  Test_accuracy: '+ str(acc1))
    
    ## CI test for the current phase
    correct_cis = 0
    totals = 0
    task_pred_cors = 0

    num_task_ = task + 1
    for task in range(num_task_):
        # print('Testing Task :---'+str(task))
        test_loader = task_data[task][1]
        correct_ci, total, task_pred_cor = inferecne(test_loader, task, num_task_, net)
        
        correct_cis += correct_ci
        totals += total
        task_pred_cors += task_pred_cor.item()

    # pdb.set_trace()
    task_acc_ = correct_cis / totals * 100.
    task_pred_acc_ = task_pred_cors / totals * 100.

    ## report CI acc. and task-id acc.
    ci_acc_list.append(task_acc_)
    print('Total tasks: {}, CIL Acc: {:.2f}, Task-id Cls. Acc.:  {:.2f}'.format(num_task_, task_acc_, task_pred_acc_))



print(ci_acc_list)
print('Average Incremental Accuracy: {:.2f}'.format(np.mean(ci_acc_list)))


