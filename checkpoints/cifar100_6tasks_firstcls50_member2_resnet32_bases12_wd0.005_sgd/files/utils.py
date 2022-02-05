import torch
import numpy as np
import re, random, collections
import os
import sys
import json
import copy
from datetime import datetime
from models.Conv_DCFE import *

def save_model(args,task,acc,model):
    print('Saving..')
    statem = {
        'net': model.state_dict(),
        'acc': acc,
    }
    fname=args.model_path
    os.makedirs(args.model_path, exist_ok=True)
    torch.save(statem, fname+'/ckpt_task'+str(task)+'.pth')


def load_model(args,task,model):
    fname=args.model_path
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    print(fname+'/ckpt_task'+str(task)+'.pth')
    checkpoint = torch.load(fname+'/ckpt_task'+str(task)+'.pth')
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    return best_acc

def load_model_resume(args,task,model):
    ## model load used for resume training
    fname=args.model_path
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    print(fname+'/ckpt_task'+str(task)+'.pth')
    checkpoint = torch.load(fname+'/ckpt_task'+str(task)+'.pth')['net']
    checkpoint = {k: v for k, v in checkpoint.items() if not ((f'branch_list.' in k) and ('coef' in k))}
    model.load_state_dict(checkpoint)



def load_past(args, task, model, copy_head=False, copy_first=False):
    """
      used for dcf models, load branch[task-1] params for branch[task]
    """
    copy_id = 0 if copy_first else task-1
    pre_ckpt_file = os.path.join(args.model_path, f'ckpt_task{copy_id}.pth')
    ckpt = torch.load(pre_ckpt_file)['net']
    # ckpt = {k: v for k, v in ckpt.items() if not (('branch_list' in k) and ('coef' in k))}
    
    ckpt_new = copy.deepcopy(ckpt)
    ## copy previous branch's param for the new branch, including the head
    pre_task_id = task -1
    for k, v in ckpt.items():
        if f'branch_list.{pre_task_id}' in k:
            ckpt_new[k.replace(f'branch_list.{pre_task_id}', f'branch_list.{task}')] = copy.deepcopy(v)
        
        if f'heads.{pre_task_id}' in k:
            if copy_head:
                ckpt_new[k.replace(f'heads.{pre_task_id}', f'heads.{task}')] = copy.deepcopy(v)
            else:
                cur_dict = model.state_dict()
                ckpt_new[k.replace(f'heads.{pre_task_id}', f'heads.{task}')] = cur_dict[k.replace(f'heads.{pre_task_id}', f'heads.{task}')]
            
    ## delete 'coef' attribute in all CONV_DCFDE layers in the model
    for task_i in range(task):
        branch_ = model.branch_list[task_i]
        for m in branch_.modules():
            # if isinstance(m, Conv_DCFDE) and hasattr(m, 'coef'):
                # delattr(m, 'coef')
            if hasattr(m, 'coef'):
                delattr(m, 'coef')

    ## delete 'coef' attribute in all CONV_DCFDE layers in the checkpoint
    ckpt_new = {k: v for k, v in ckpt_new.items() if not ((f'branch_list.' in k) and ('coef' in k))}
    model.load_state_dict(ckpt_new)



def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    
    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)