import torch
import numpy as np
import os
import random


class DataLoader(object):
    def __init__(self, path):
        self.path = path
        self.datas = []
        self.label_ind = []
        self.label_val = []
        for d in os.listdir('{}/ground_truth'.format(path)):
            inds, vals = [], []
            with open('{}/ground_truth/{}'.format(path, d), 'r') as f:
                for l in f:
                    ls = l.split()
                    inds.append(int(ls[0]))
                    vals.append(float(ls[1]))
            if len(inds) == 0:
                continue
            self.label_ind.append(inds)
            self.label_val.append(vals)
            with open('{}/tb_mem_cnt/{}'.format(path, d), 'r') as f:
                l_x = list(map(float, f.read().split()))
                x = torch.tensor(l_x, dtype=torch.float32)
                x = x.reshape((len(l_x), 1))
            self.datas.append(x)
        self.train_cut = int(len(self.datas) * .9)
    
    def next(self, train=True):
        if train:
            idx = random.randint(0, self.train_cut - 1)
        return self.get(idx)

    def get(self, idx):
        return self.datas[idx], self.label_ind[idx], self.label_val[idx]

