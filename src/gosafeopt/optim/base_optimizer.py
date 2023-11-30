import torch
import gosafeopt
from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.aquisitions.go_safe_opt import GoSafeOpt
from gosafeopt.tools.points import random, uniform
from gosafeopt.tools.logger import Logger
from abc import abstractmethod
from gosafeopt.tools.misc import singleton
from typing import Optional
from torch import Tensor
from torch.distributions import MultivariateNormal
from typing import Tuple
from gosafeopt.optim.safe_set import SafeSet


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
        SafeSet.configure(max_global_steps_without_progress, max_global_steps_without_progress_tolerance)

    @abstractmethod
    def optimize(self, step: int = 0) -> Tuple[Tensor, Tensor]:
        pass

    def get_initial_params(self, mode: str):
        override_mode = self.aquisition.override_set_initialization()
        mode = override_mode if isinstance(override_mode, str) else mode
        X: Tensor
        if mode == "random":
            X = random(self.domain_start, self.domain_end, self.set_size, self.dim_params)
            X = torch.hstack([X, self.context.repeat(len(X), 1)])

        elif mode == "uniform":
            X = uniform(self.domain_start, self.domain_end, self.set_size, self.dim_params)
            X = torch.hstack([X, self.context.repeat(len(X), 1)])

        elif mode == "safe" or "safe_all":
            N = self.set_size

            safe_set = SafeSet.get_current_safe_set()
            #
            # Use initial safe point as seed
            if safe_set is None or len(safe_set) == 0:
                safe_set = self.aquisition.data.train_x[-1:].to(gosafeopt.device)
            else:
                # TODO: why is this needed? Should already be on correct device.
                for i in range(len(SafeSet.safe_sets)):
                    SafeSet.safe_sets[i] = SafeSet.safe_sets[i].to(gosafeopt.device)
                safe_set = torch.vstack(SafeSet.safe_sets) if mode == "safe_all" else SafeSet.get_current_safe_set()

            # Sample at most N points from Safeset
            if safe_set.shape[0] >= N:
                X = safe_set[torch.randint(0, safe_set.shape[0], (N,))]
            # If |safesest| < N sample randomly around safeset
            else:
                distribution = MultivariateNormal(
                    safe_set.mean(axis=0), 1e-3 * torch.eye(safe_set.shape[1], device=gosafeopt.device)
                )
                X = distribution.rsample([N])
                X[:, -self.dim_context :] = self.context.repeat(N, 1)
                X[: len(safe_set)] = safe_set

        else:
            raise RuntimeError("Set init not defined")

        return X.to(gosafeopt.device)

    def optimize_steps(self) -> Tensor | Tensor:
        X, reward = None, None
        rewardMax = -1e10
        rewardStage = 0

        for step in range(self.aquisition.steps):
            [X_step, reward_step] = self.optimize(step)
            # Some steps are internal. Such as calculating global ucb...
            if not self.aquisition.is_internal_step(step):
                X, reward = (
                    (X_step, reward_step)
                    if X is None
                    else (torch.vstack([X, X_step]), torch.vstack([reward, reward_step]))
                )

                if reward_step.max() > rewardMax:
                    rewardMax = reward_step.max()
                    rewardStage = step

        Logger.info(f"MultiStageAquisition is taken from step {rewardStage}")
        return [X, reward]

    def next_params(self):
        [X, reward] = self.optimize_steps()

        next_param = X[torch.argmax(reward)]
        reward = reward.max()

        # TODO: make this enum
        if self.set_init == "safe":
            SafeSet.update_safe_set(X, self.aquisition)

        if not self.aquisition.has_safe_points(X):
            Logger.warn("Could not find safe set")

        return [next_param.detach().to("cpu"), reward.detach().to("cpu")]
