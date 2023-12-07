from abc import abstractmethod
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal

import gosafeopt
from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.optim.safe_set import SafeSet
from gosafeopt.tools.data import Data
from gosafeopt.tools.logger import Logger
from gosafeopt.tools.points import random, uniform


class BaseOptimizer:
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
        context: Optional[Tensor],
    ):
        self.domain_start = domain_start
        self.domain_end = domain_end
        self.set_size = set_size
        self.max_global_steps_without_progress = max_global_steps_without_progress
        self.max_global_steps_without_progress_tolerance = max_global_steps_without_progress_tolerance
        self.dim_params = dim_params
        self.dim_context = dim_context
        self.aquisition = aquisition
        self.context = context
        self.set_init = set_init
        self.data = data
        SafeSet.configure(max_global_steps_without_progress, max_global_steps_without_progress_tolerance)

    @abstractmethod
    def optimize(self, step: int = 0) -> Tuple[Tensor, Tensor]:
        pass

    def get_initial_params(self, mode: str):
        if self.data.train_x is None:
            raise Exception("Training data is empty")

        override_mode = self.aquisition.override_set_initialization()
        mode = override_mode if isinstance(override_mode, str) else mode
        x: Tensor
        if mode == "random":
            x = random(self.domain_start, self.domain_end, self.set_size, self.dim_params)
            if self.context is not None:
                x = torch.hstack([x, self.context.repeat(len(x), 1)])

        elif mode == "uniform":
            x = uniform(self.domain_start, self.domain_end, self.set_size, self.dim_params)
            if self.context is not None:
                x = torch.hstack([x, self.context.repeat(len(x), 1)])

        elif mode == "safe" or mode == "safe_all":
            n = self.set_size

            safe_set = SafeSet.get_current_safe_set()
            #
            # Use initial safe point as seed
            if safe_set is None or len(safe_set) == 0:
                safe_set = self.data.train_x[-1:].to(gosafeopt.device)
            else:
                # TODO: why is this needed? Should already be on correct device.
                for i in range(len(SafeSet.safe_sets)):
                    SafeSet.safe_sets[i] = SafeSet.safe_sets[i].to(gosafeopt.device)
                safe_set = torch.vstack(SafeSet.safe_sets) if mode == "safe_all" else SafeSet.get_current_safe_set()

            if safe_set is None:
                raise Exception("Safe set is none")

            # Sample at most N points from Safeset
            if safe_set.shape[0] >= n:
                x = safe_set[torch.randint(0, safe_set.shape[0], (n,))]
            # If |safesest| < N sample randomly around safeset
            else:
                distribution = MultivariateNormal(
                    safe_set.mean(axis=0),  # type: ignore
                    1e-3 * torch.eye(safe_set.shape[1], device=gosafeopt.device),
                )
                x = distribution.rsample(torch.Size([n]))
                if self.context is not None:
                    x[:, -self.dim_context :] = self.context.repeat(n, 1)
                x[: len(safe_set)] = safe_set

        else:
            raise RuntimeError("Set init not defined")

        return x.to(gosafeopt.device)

    def optimize_steps(self) -> Tuple[Tensor, Tensor]:
        x, reward = None, None
        reward_max = -1e10
        reward_stage = 0

        for step in range(self.aquisition.steps):
            [x_step, reward_step] = self.optimize(step)
            # Some steps are internal. Such as calculating global ucb...
            if not self.aquisition.is_internal_step(step):
                x, reward = (
                    (x_step, reward_step)
                    if x is None or reward is None
                    else (torch.vstack([x, x_step]), torch.vstack([reward, reward_step]))
                )

                if reward_step.max() > reward_max:
                    reward_max = reward_step.max()
                    reward_stage = step

        Logger.info(f"MultiStageAquisition is taken from step {reward_stage}")
        return x, reward

    def next_params(self):
        [x, reward] = self.optimize_steps()

        next_param = x[torch.argmax(reward)]
        reward = reward.max()

        # TODO: make this enum
        if self.set_init == "safe":
            SafeSet.update_safe_set(x, self.aquisition)

        if not self.aquisition.has_safe_points(x):
            Logger.warn("Could not find safe set")

        return [next_param.detach().to("cpu"), reward.detach().to("cpu")]
