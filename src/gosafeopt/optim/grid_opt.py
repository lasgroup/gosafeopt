from gpytorch.kernels.index_kernel import Optional
from torch import Tensor

from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.optim.base_optimizer import BaseOptimizer
from gosafeopt.tools.data import Data


class GridOpt(BaseOptimizer):
    def __init__(
        self,
        aquisition: BaseAquisition,
        domain_start: Tensor,
        domain_end: Tensor,
        max_global_steps_without_progress_tolerance: int,
        max_global_steps_without_progress: int,
        set_size: int,
        dim_params: int,
        dim_context: int,
        set_init: str,
        data: Data,
        context: Optional[Tensor] = None,
    ):
        super().__init__(
            aquisition,
            domain_start,
            domain_end,
            max_global_steps_without_progress_tolerance,
            max_global_steps_without_progress,
            set_size,
            dim_params,
            dim_context,
            set_init,
            data,
            context,
        )

    def optimize(self, step: int = 0):
        x = self.get_initial_params(self.set_init)

        loss = self.aquisition.evaluate(x, step)

        return [x, loss]
