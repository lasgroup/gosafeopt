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
        if c["set_init"] == "uniform" and not is_square(c["set_size"]):
            raise Exception("Set size should be square (nxn)")

    def optimize(self):
        i = 0
        # N Restarts if no safe set is found
        while i == 0 or (i < self.c["swarmopt_n_restarts"] and not self.aquisition.hasSafePoints(x)):
            x = self.getInitPoints(self.c["set_init"], self.c["append_train_set"])
            p = copy.deepcopy(x)

            res = self.evaluate_aquisition(x)

            fBest = res.max()
            pBest = x[torch.argmax(res)]

            v = rand2n_torch(-np.abs(
                torch.tensor(self.c["domain_end"])-torch.tensor(self.c["domain_start"])),
                np.abs(torch.tensor(self.c["domain_end"])-torch.tensor(self.c["domain_start"])),
                x.shape[0], self.c["dim"]).to(gosafeopt.device)/10

            inertia_scale = self.c["swarmopt_w"]

            # Swarmopt
            for _ in range(self.c["swarmopt_n_iterations"]):
                # Update swarm velocities
                # print(x.mean(axis=0))
                r_p = rand2n_torch(torch.tensor([0]).repeat(self.c["dim"]), torch.tensor(
                    [1]).repeat(self.c["dim"]), x.shape[0], self.c["dim"]).to(gosafeopt.device)
                r_g = rand2n_torch(torch.tensor([0]).repeat(self.c["dim"]), torch.tensor(
                    [1]).repeat(self.c["dim"]), x.shape[0], self.c["dim"]).to(gosafeopt.device)
                v = inertia_scale*v + self.c["swarmopt_p"] * \
                    r_p * (p-x) + self.c["swarmopt_g"]*r_g*(pBest-x)
                inertia_scale *= 0.95

                # Update swarm position
                if self.c["dim_context"] > 0:
                    v[:, -self.c["dim_context"]:] = 0

                v = clamp2dTensor(v, torch.tensor([-10*self.c["swarmopt_w"]], device = gosafeopt.device).repeat(v.shape[1]), torch.tensor([10*self.c["swarmopt_w"]], device = gosafeopt.device).repeat(v.shape[1]))
                x += v
                x = clamp2dTensor(x, torch.tensor(
                    self.c["domain_start"], device=gosafeopt.device), torch.tensor(self.c["domain_end"], device=gosafeopt.device))

                resTmp = self.evaluate_aquisition(x)

                mask = resTmp > res
                p[mask] = x[mask]
                res[mask] = resTmp[mask]

                if res.max() > fBest:
                    fBest = res.max()
                    pBest = x[torch.argmax(res)]

                if  torch.cuda.is_available(): 
                    torch.cuda.empty_cache()

            i += 1

        return [x, res]
