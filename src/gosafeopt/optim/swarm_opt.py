from gosafeopt.optim.base_optimizer import BaseOptimizer
from gosafeopt.tools.rand import rand2n_torch
from gosafeopt.tools.math import clamp2dTensor
import torch
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
import gosafeopt


class SwarmOpt(BaseOptimizer):
    def __init__(self, aquisition, c, context=None):
        super().__init__(aquisition, c, context)

        def is_square(n):
            root = math.isqrt(n)
            return n == root * root

        # TODO: rethink this
        if c["set_init"] == "uniform" and not is_square(c["set_size"]):
            raise Exception("Set size should be square (nxn)")

    def optimize(self, step: int = 0):
        i = 0
        # N Restarts if no safe set is found
        while i == 0 or (i < self.config["swarmopt_n_restarts"] and not self.aquisition.has_safe_points(x)):
            x = self.get_initial_params(self.config["set_init"])
            p = copy.deepcopy(x)

            res = self.aquisition.evaluate(x, step)

            fBest = res.max()
            pBest = x[torch.argmax(res)]

            v = (
                rand2n_torch(
                    -np.abs(torch.tensor(self.config["domain_end"]) - torch.tensor(self.config["domain_start"])),
                    np.abs(torch.tensor(self.config["domain_end"]) - torch.tensor(self.config["domain_start"])),
                    x.shape[0],
                    self.config["dim"],
                ).to(gosafeopt.device)
                / 10
            )

            inertia_scale = self.config["swarmopt_w"]

            # Swarmopt
            for _ in range(self.config["swarmopt_n_iterations"]):
                # Update swarm velocities
                # print(x.mean(axis=0))
                r_p = rand2n_torch(
                    torch.tensor([0]).repeat(self.config["dim"]),
                    torch.tensor([1]).repeat(self.config["dim"]),
                    x.shape[0],
                    self.config["dim"],
                ).to(gosafeopt.device)
                r_g = rand2n_torch(
                    torch.tensor([0]).repeat(self.config["dim"]),
                    torch.tensor([1]).repeat(self.config["dim"]),
                    x.shape[0],
                    self.config["dim"],
                ).to(gosafeopt.device)
                v = (
                    inertia_scale * v
                    + self.config["swarmopt_p"] * r_p * (p - x)
                    + self.config["swarmopt_g"] * r_g * (pBest - x)
                )
                inertia_scale *= 0.95

                # Update swarm position
                if self.config["dim_context"] > 0:
                    v[:, -self.config["dim_context"] :] = 0

                v = clamp2dTensor(
                    v,
                    torch.tensor([-10 * self.config["swarmopt_w"]], device=gosafeopt.device).repeat(v.shape[1]),
                    torch.tensor([10 * self.config["swarmopt_w"]], device=gosafeopt.device).repeat(v.shape[1]),
                )
                x += v
                x = clamp2dTensor(
                    x,
                    torch.tensor(self.config["domain_start"], device=gosafeopt.device),
                    torch.tensor(self.config["domain_end"], device=gosafeopt.device),
                )

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
