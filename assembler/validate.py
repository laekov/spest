import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from model.estimator import Estimator
from dataloader import DataLoader
import config


def main():
    dataset= DataLoader(config.data_path)
    model = Estimator(config.input_size, config.hidden_size)
    model.eval()
    if config.load_path is not None and os.path.exists(config.load_path):
        model.load_state_dict(torch.load(config.load_path))
    loss_accum = None
    best_loss = None
    for i in range(dataset.train_cut, len(dataset.datas)):
        data, idxs, label = dataset.get(i)
        out = model(data, idxs)
        for l, o in zip(label.detach().numpy(), out.detach().numpy()):
            print('{}\t{}'.format(l, o))

if __name__ == '__main__':
    main()

