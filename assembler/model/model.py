import torch.nn as nn
from .estimator import Estimator
from .comparator import Comparator
import config


class PerfCompare(nn.Module):
    def __init__(self):
        super(PerfCompare, self).__init__()
        self.est = Estimator(config.input_size, config.hidden_size)
        self.comp = Comparator(config.output_size)

