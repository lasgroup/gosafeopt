import copy
import math
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

import gosafeopt
from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.optim.base_optimizer import BaseOptimizer
from gosafeopt.tools.math import clamp2dTensor
from gosafeopt.tools.rand import rand2n_torch


class SwarmOpt(BaseOptimizer):
    def __init__(
        self,
        aquisition: BaseAquisition,
        domain_start: Tensor,
        domain_end: Tensor,
        p: float,
        w: float,
        g: float,
        max_global_steps_without_progress_tolerance: int,
        max_global_steps_without_progress: int,
        set_size: int,
        dim_params: int,
        dim_context: int,
        set_init: str,
        n_restarts: int,
        n_iterations: int,
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
            context,
        )
        self.n_restarts = n_restarts
        self.n_iterations = n_iterations
        self.p = p
        self.g = g
        self.w = w

        def is_square(n):
            root = math.isqrt(n)
            return n == root * root

        # TODO: rethink this
        if self.set_init == "uniform" and not is_square(self.set_size):
            raise Exception("Set size should be square (nxn)")

    def optimize(self, step: int = 0):
        i = 0
        # N Restarts if no safe set is found
        while i == 0 or (i < self.n_restarts and not self.aquisition.has_safe_points(x)):
            x = self.get_initial_params(self.set_init)
            p = copy.deepcopy(x)

            res = self.aquisition.evaluate(x, step)

            fBest = res.max()
            pBest = x[torch.argmax(res)]

            v = (
                rand2n_torch(
                    -np.abs(self.domain_end - self.domain_start),
                    np.abs(self.domain_end - self.domain_start),
                    x.shape[0],
                    self.dim_params + self.dim_context,
                ).to(gosafeopt.device)
                / 10
            )

            inertia_scale = self.w
            dim = self.dim_context + self.dim_params

            # Swarmopt
            for _ in range(self.n_iterations):
                # Update swarm velocities
                r_p = rand2n_torch(
                    torch.tensor([0]).repeat(dim),
                    torch.tensor([1]).repeat(dim),
                    x.shape[0],
                    dim,
                ).to(gosafeopt.device)
                r_g = rand2n_torch(
                    torch.tensor([0]).repeat(dim),
                    torch.tensor([1]).repeat(dim),
                    x.shape[0],
                    dim,
                ).to(gosafeopt.device)
                v = inertia_scale * v + self.p * r_p * (p - x) + self.g * r_g * (pBest - x)
                inertia_scale *= 0.95

                # Update swarm position
                if self.dim_context > 0:
                    v[:, -self.dim_context :] = 0

                v = clamp2dTensor(
                    v,
                    torch.tensor([-10 * self.w], device=gosafeopt.device).repeat(v.shape[1]),
                    torch.tensor([10 * self.w], device=gosafeopt.device).repeat(v.shape[1]),
                )
                x += v
                x = clamp2dTensor(x, self.domain_start, self.domain_end)

                resTmp = self.aquisition.evaluate(x, step)

                mask = resTmp > res
                p[mask] = x[mask]
                res[mask] = resTmp[mask]

                if res.max() > fBest:
                    fBest = res.max()
                    pBest = x[torch.argmax(res)]

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            i += 1

        return [x, res]
