import torch
import torch.nn as nn
import torch.nn.functional as F


from .rnn import RNN
from config import output_size

class Estimator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Estimator, self).__init__()
        self.RNN = RNN(input_size, hidden_size, output_size)
        self.fc1 = nn.Linear(output_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, out_idx):
        h = self.RNN.init_hidden()
        outs = []
        j = 0
        for i in range(out_idx[-1] + 1):
            o, h = self.RNN(x[i], h)
            if i == out_idx[j]:
                outs.append(o)
                j += 1
        outs = torch.stack(outs)
        outs = self.fc1(outs)
        outs = F.relu(outs)
        outs = self.fc2(outs).reshape((len(out_idx), ))
        return outs
