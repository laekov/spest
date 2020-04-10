import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random

from model.model import PerfCompare
from dataloader import DataLoader
from emave import ExpMovAve
import config


def main():
    dataset = DataLoader(config.data_path)
    model = PerfCompare()

    if config.load_path is not None and os.path.exists(config.load_path):
        model.load_state_dict(torch.load(config.load_path))

    if config.cuda:
        model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    loss_accum = ExpMovAve()
    accu_accum = ExpMovAve()
    best_loss = None

    for i in range(config.train_iters):
        labels = []
        outs = []
        for _ in range(config.batch_size):
            data, idxs, label = dataset.next()
            data = data.cuda()
            out = model.est(data, idxs)
            labels = labels + label
            outs.append(out)

        outs = torch.cat(outs, dim=0)
        
        cmpa, cmpb = [], []
        answer = []
        for _ in range(config.comp_size):
            ia, ib = random.randint(0, len(labels) - 1), random.randint(0, len(labels) - 1)
            cmpa.append(outs[ia])
            cmpb.append(outs[ib])
            ans = 2
            if labels[ia] < labels[ib]:
                ans = 0
            elif labels[ia] > labels[ib]:
                ans = 1
            answer.append(ans)

        cmpa = torch.stack(cmpa)
        cmpb = torch.stack(cmpb)
        answer = torch.tensor(answer)
        if config.cuda:
            answer = answer.cuda()
        cmp_res = model.comp(cmpa, cmpb)
        loss = F.cross_entropy(cmp_res, answer)
        if torch.isnan(loss):
            raise ValueError('Loss becomes NaN! Last datasets are: \n{}'.format(
                dataset.get_last_records(64)))
        prediction = cmp_res.argmax(dim=1)
        accu = (prediction == answer).int().sum() / float(answer.shape[0])

        loss.backward()
        optim.step()

        loss_accum.add(loss.item())
        accu_accum.add(accu.item())
        if i % 10 == 0:
            print('Iteration {} loss {:.5} accuracy {:.4} accum accuracy {:.4}'.format(
                i, loss_accum.value, accu.item(), accu_accum.value))
            if False: #best_loss is None or loss_accum.value < best_loss:
                best_loss = loss_accum.value
                print('Saving')
                torch.save(model.state_dict(), 'ckpt/1.pt')


if __name__ == '__main__':
    main()

