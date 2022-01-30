import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
import pdb

Class_100_List = list(range(100))


file_ = open('/home/mzc/Continual_Learning/EFT/Datasets/imagenet100_s1993.txt', 'r')
Class_100_List_1993 = [f.strip() for f in file_.readlines()]

class ImageNet100(ImageFolder):
    def __init__(self, root, transform, list_used='1993'):
        super().__init__(root, transform)
        ## selet the 100 classes
        
        # pdb.set_trace()
        if list_used == 'f100':
            self.samples = [sample for sample in self.samples if sample[1] in Class_100_List]
        elif list_used == '1993':
            class_list_ = [self.class_to_idx[k] for k in Class_100_List_1993]
            self.samples = [sample for sample in self.samples if sample[1] in class_list_]
        else:
            raise KeyError('...')
        self.targets = [s[1] for s in self.samples]

