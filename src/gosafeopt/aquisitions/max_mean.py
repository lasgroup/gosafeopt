from typing import Optional

import torch
from torch import Tensor

from gosafeopt.aquisitions.base_aquisition import BaseAquisition


class MaxMean(BaseAquisition):
    def __init__(
        self,
        dim_obs: int,
        scale_beta: float,
        beta: float,
        context: Optional[Tensor] = None,
    ):
        super().__init__(dim_obs, scale_beta, beta, context)

    def evaluate(self, x: Tensor, step: int = 0) -> Tensor:  # noqa: ARG002
        posterior = self.model_posterior(x)
        mean = posterior.mean.reshape(-1, self.dim_obs)
        var = posterior.variance.reshape(-1, self.dim_obs)

        l = mean - self.scale_beta * torch.sqrt(self.beta * var)  # noqa: E741

        safe_set = torch.all(l[:, 1:] > self.fmin[1:], axis=1)  # type: ignore
        mean[~safe_set] = -1e10

        return mean[:, 0]
