import torch
import torch.nn as nn
import torch.nn.functional as F
import sys


from .rnn import RNN
from config import output_size


class Estimator(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Estimator, self).__init__()
        # self.rnn = RNN(input_size, hidden_size, output_size)
        # self.rnn = nn.RNNCell(input_size, hidden_size)
        self.conv1 = torch.nn.Conv1d(input_size, hidden_size, 9, padding=4)
        self.conv2 = torch.nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.hidden_size = hidden_size
        self.cnt = 0

    def forward(self, x0, out_idx):
        self.cnt += 1
        outs = []
        for oi in out_idx:
            x = torch.stack([x0[:oi]]).transpose(1, 2)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = x.sum(dim=2)
            outs.append(x)
        outs = torch.cat(outs, dim=0)

        outs = self.fc1(outs)
        return outs
