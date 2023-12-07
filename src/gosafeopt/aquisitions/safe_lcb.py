from typing import Optional

import torch
from torch import Tensor

from gosafeopt.aquisitions.base_aquisition import BaseAquisition


class SafeLCB(BaseAquisition):
    def __init__(
        self,
        dim_obs: int,
        scale_beta: float,
        beta: float,
        context: Optional[Tensor] = None,
    ):
        super().__init__(dim_obs, scale_beta, beta, context)

    def evaluate(self, x: Tensor):
        posterior = self.model_posterior(x)
        l, _ = self.get_confidence_interval(posterior)  # noqa: E741

        safe_set = torch.all(l[:, 1:] > self.fmin[1:], axis=1)  # type: ignore
        l[~safe_set] = -1e10

        return l[:, 0]
