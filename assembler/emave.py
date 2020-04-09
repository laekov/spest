from config import mave_weight


class ExpMovAve(object):
    def __init__(self, weight=mave_weight):
        self.weight = weight
        self.value = None

    def add(self, v):
        if self.value is None:
            self.value = v
        else:
            self.value = self.value * self.weight + v * (1. - self.weight)
        
    def __float__(self):
        if self.value is not None:
            return self.value
        return 0.

    def __repr__(self):
        if self.value is None:
            return 'N/A'
        return str(self.value)
