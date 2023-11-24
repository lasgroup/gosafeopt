import torch

from gosafeopt.aquisitions.base_aquisition import BaseAquisition


class MaxMean(BaseAquisition):
    def __init__(self, model, c, context=None, data=None):
        super().__init__(model, c, context, data)

    def evaluate(self, X):
        mean = X.mean.reshape(-1, self.config["dim_obs"])
        var = X.variance.reshape(-1, self.config["dim_obs"])

        l = mean - self.config["scale_beta"] * torch.sqrt(self.config["beta"] * var)

        S = torch.all(l[:, 1:] > self.fmin[1:], axis=1)
        mean[~S] = -1e10

        return mean[:, 0]
