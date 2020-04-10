import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=0)

    def forward(self, x, h, need_out=False):
        c = torch.cat((x, h), dim=0)
        h = self.i2h(c)
        if need_out:
            o = self.i2o(c)
            o = self.softmax(o)
        else:
            o = None
        return o, h

    def init_hidden(self):
        return torch.zeros(self.hidden_size, device=self.i2h.weight.device)

