from typing import Optional

from torch import Tensor

from gosafeopt.aquisitions.base_aquisition import BaseAquisition


class UCB(BaseAquisition):
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
        _, ucb = self.get_confidence_interval(posterior)

        loss_perf = ucb[:, 0]

        return loss_perf
