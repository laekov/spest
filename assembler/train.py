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
    if config.load_path is not None and os.path.exists(config.load_path):
        model.load_state_dict(torch.load(config.load_path))
    if config.cuda:
        model.cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_accum = None
    best_loss = None
    for i in range(config.train_iters):
        data, idxs, label = dataset.next()
        data, label = data.cuda(), label.cuda()
        out = model(data, idxs)
        loss = F.l1_loss(out, label) / len(idxs)
        if loss_accum is None:
            loss_accum = loss.item()
        else:
            loss_accum = config.mave_weight * loss_accum + (1. - config.mave_weight) * loss.item()
        loss.backward()
        optim.step()
        if i % 100 == 0:
            print('Iteration {} loss {}'.format(i, loss_accum))
            if best_loss is None or loss_accum < best_loss:
                best_loss = loss_accum
                print('Saving')
                torch.save(model.state_dict(), 'ckpt/1.pt')



if __name__ == '__main__':
    main()

