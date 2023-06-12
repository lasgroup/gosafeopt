from gosafeopt.optim.base_optimizer import BaseOptimizer
import torch


class GridOpt(BaseOptimizer):

    def __init__(self, aquisition, c, context=None):
        super().__init__(aquisition, c, context)

    def optimize(self):
        X = self.getInitPoints(self.c["set_init"], self.c["append_train_set"])

        loss = self.evaluate_aquisition(X)

        return [X, loss]
