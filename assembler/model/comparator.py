import torch
import torch.nn as nn
import torch.nn.functional as F


class Comparator(nn.Module):
    def __init__(self, input_size):
        super(Comparator, self).__init__()
        self.fc = nn.Linear(input_size * 2, 3)

    def forward(self, x, y):
        x = F.relu(x)
        y = F.relu(y)
        o = self.fc(torch.cat((x, y), dim=1))
        return o
