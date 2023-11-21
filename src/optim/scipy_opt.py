from botorch.generation.gen import gen_candidates_scipy
from botorch.optim.initializers import gen_batch_initial_conditions
import numpy as np
from botorch.optim import optimize_acqf
import torch
from gosafeopt.optim.base_optimizer import BaseOptimizer

# TODO maybe fix again
class ScipyOpt(BaseOptimizer):

    def __init__(self, aquisition, c, context=None):
        super().__init__(aquisition, c, context)

    def optimize(self):
        bounds = torch.vstack([torch.tensor(self.c["domain_start"]), torch.tensor(self.c["domain_end"])])
        # xInit = self.getInitPoints()
        xInit = gen_batch_initial_conditions(
            self.aquisition, bounds, q=1, num_restarts=self.c["set_size"], raw_samples=self.c["set_size"]
        )
        batch_candidates, batch_acq_values = gen_candidates_scipy(
            initial_conditions=xInit,
            acquisition_function=self.aquisition,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1],
        )

        idx = torch.argmax(batch_acq_values)
        return [batch_candidates.detach().squeeze()[idx], batch_acq_values[idx].detach()]
