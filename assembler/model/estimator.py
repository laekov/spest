import torch
import torch.nn as nn
import torch.nn.functional as F


from .rnn import RNN
from config import output_size

class Estimator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Estimator, self).__init__()
        self.RNN = RNN(input_size, hidden_size, output_size)
        self.fc1 = nn.Linear(output_size, output_size)

    def forward(self, x, out_idx):
        h = self.RNN.init_hidden()
        outs = []
        j = 0
        for i in range(out_idx[-1] + 1):
            need_out = (i == out_idx[j])
            o, h = self.RNN(x[i], h, need_out=need_out)
            if need_out:
                outs.append(o)
                j += 1
        outs = torch.stack(outs)
        outs = F.softmax(outs, dim=1)
        outs = self.fc1(outs)
        return outs
