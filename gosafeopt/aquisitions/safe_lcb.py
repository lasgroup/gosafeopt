
from gosafeopt.aquisitions.base_aquisition import BaseAquisition
import torch


class SafeLCB(BaseAquisition):
    def __init__(self, model, c, context=None, data=None):
        super().__init__(model, c, context, data)

    def evaluate(self, X):
        l, _ = self.getBounds(X)

        S = torch.all(l[:, 1:] > self.fmin[1:], axis=1)
        l[~S] = -1e10

        return l[:, 0]
