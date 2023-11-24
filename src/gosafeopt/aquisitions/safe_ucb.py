from gosafeopt.aquisitions.base_aquisition import BaseAquisition
import torch


class SafeUCB(BaseAquisition):
    def __init__(self, model, c, context=None, data=None):
        super().__init__(model, c, context, data)

    def evaluate(self, X):
        l, u = self.get_confidence_interval(X)
        values = u[:, 0]

        slack = l - self.fmin

        if self.config["use_soft_penalties"]:
            values += self.soft_penalty(slack)
        else:
            S[:] = torch.all(l[:, 1:] > self.fmin[1:], axis=1)
            values[~S] = -1e10
        return values
