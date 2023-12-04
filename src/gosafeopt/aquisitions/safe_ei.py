from typing import Optional

import torch
from scipy.stats import norm
from torch import Tensor

from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.tools.data import Data


class SafeEI(BaseAquisition):
    def __init__(
        self,
        dim_obs: int,
        scale_beta: float,
        beta: float,
        data: Data,
        context: Optional[Tensor] = None,
    ):
        super().__init__(dim_obs, scale_beta, beta, context)
        self.data = data

    def evaluate(self, x: Tensor):
        if self.data.train_x is None:
            raise Exception("Training data is empty")

        posterior = self.model_posterior(x)
        l, _ = self.get_confidence_interval(posterior)  # noqa: E741

        xi = 0.01

        mu = posterior.mean.reshape(-1, self.dim_obs)[:, 0]
        sigma = posterior.variance.reshape(-1, self.dim_obs)[:, 0]

        mean_sample = self.model_posterior(self.data.train_x).mean.detach()[:, 0]
        mu_sample_opt = torch.max(mean_sample)

        imp = mu - mu_sample_opt - xi
        z = (imp / sigma).detach().numpy()
        ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
        ei[sigma == 0.0] = 0.0

        slack = l - self.fmin

        ei += self.soft_penalty(slack)

        return ei
