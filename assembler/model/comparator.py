import torch
import torch.nn as nn
import torch.nn.functional as F


class Comparator(nn.Module):
    def __init__(self, input_size):
        super(Comparator, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x, y):
        i = torch.cat((x, y), dim=1)
        # i = F.softmax(i, dim=1)
        o = self.fc1(i)
        o = F.tanh(o)
        o = self.fc2(o)
        return o
