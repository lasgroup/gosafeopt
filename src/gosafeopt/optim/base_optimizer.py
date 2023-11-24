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


# TODO: self.i changing sets should should not be handled here
@singleton
class SafeSet(object):
    def __init__(self, config: dict):
        self.config = config
        self.reset()

    def reset(self):
        self.safe_sets = []
        self.current_safe_set = 0
        self.best_sage_set = 0
        self.y_min = -1e10
        self.global_y_min = -1e10
        self.i = 0

    def get_current_safe_set(self) -> Optional[Tensor]:
        if len(self.safe_sets) == 0:
            return None
        else:
            return self.safe_sets[self.current_safe_set]

    def update_safe_set(self, X: Tensor, aquisition: BaseAquisition):
        new_safe_set = X[aquisition.safe_set(X)]  # New params considered safe
        safe_set = self.get_current_safe_set()
        if safe_set is not None:
            # Remove parameters considered unsafe under the updated model
            still_safe = aquisition.safe_set(safe_set)
            self.filter_safe_set(still_safe)  # Remove unsafe points
            self.add_to_current_safe_set(new_safe_set)
        else:
            self.add_new_safe_set(new_safe_set)

    def filter_safe_set(self, mask: Tensor):
        self.safe_sets[self.current_safe_set] = self.safe_sets[self.current_safe_set][mask]

    def add_to_current_safe_set(self, safeset: Tensor):
        safeset.to(gosafeopt.device)
        current_safe_set = self.get_current_safe_set()
        if current_safe_set is not None:
            self.safe_sets[self.current_safe_set] = torch.vstack([current_safe_set, safeset])
        else:
            self.safe_sets[self.current_safe_set] = safeset

    def add_new_safe_set(self, safeset: Tensor):
        safeset.to(gosafeopt.device)
        self.safe_sets.append(safeset)

    def change_to_latest_safe_set(self):
        self.i = 0
        self.current_safe_set = len(self.safe_sets) - 1
        self.y_min = -1e10
        Logger.info(f"BestSet: {self.best_sage_set} / CurrentSet: {self.current_safe_set}")

    def change_to_best_safe_set(self):
        self.i = 0
        self.current_safe_set = self.best_sage_set
        Logger.info(f"Changing to best set Nr. {self.best_sage_set}")

    def calculate_current_set(self, yMin):
        if self.global_y_min < yMin:
            self.global_y_min = yMin
            self.best_sage_set = self.current_safe_set
            Logger.info(f"BestSet: {self.best_sage_set}")

        if self.y_min < yMin:
            self.y_min = yMin

        if self.y_min < (
            self.config["safe_opt_max_tries_without_progress_tolerance"] * self.global_y_min
            if self.global_y_min > 0
            else (2 - self.config["safe_opt_max_tries_without_progress_tolerance"]) * self.global_y_min
        ):
            self.i += 1

        if self.i >= self.config["safe_opt_max_tries_without_progress"]:
            self.change_to_best_safe_set()


class BaseOptimizer:
    def __init__(self, aquisition: BaseAquisition, config: dict, context: Tensor):
        self.aquisition = aquisition
        self.config = config
        self.context = context

    @abstractmethod
    def optimize(self, step: int = 0) -> Tuple[Tensor, Tensor]:
        pass

    def get_initial_params(self, mode: str):
        override_mode = self.aquisition.override_set_initialization()
        mode = override_mode if isinstance(override_mode, str) else mode
        X: Tensor
        if mode == "random":
            X = random(
                self.config["domain_start"],
                self.config["domain_end"],
                self.config["set_size"],
                self.config["dim_params"],
            )
            X = torch.hstack([X, self.context.repeat(len(X), 1)])

        elif mode == "uniform":
            X = uniform(
                self.config["domain_start"],
                self.config["domain_end"],
                self.config["set_size"],
                self.config["dim_params"],
            )
            X = torch.hstack([X, self.context.repeat(len(X), 1)])

        elif mode == "safe" or "safe_all":
            N = self.config["set_size"]

            state = SafeSet(self.config)
            safe_set = state.get_current_safe_set()
            #
            # Use initial safe point as seed
            if safe_set is None or len(safe_set) == 0:
                safe_set = self.aquisition.data.train_x[-1:].to(gosafeopt.device)
            else:
                # TODO: why is this needed? Should already be on correct device.
                for i in range(len(state.safe_sets)):
                    state.safe_sets[i] = state.safe_sets[i].to(gosafeopt.device)
                safe_set = torch.vstack(safe_set.safe_sets) if mode == "safe_all" else state.get_current_safe_set()

            # Sample at most N points from Safeset
            if safe_set.shape[0] >= N:
                X = safe_set[torch.randint(0, safe_set.shape[0], (N,))]
            # If |safesest| < N sample randomly around safeset
            else:
                distribution = MultivariateNormal(
                    safe_set.mean(axis=0), 1e-3 * torch.eye(safe_set.shape[1], device=gosafeopt.device)
                )
                X = distribution.rsample([N])
                X[:, -self.config["dim_context"] :] = self.context.repeat(N, 1)
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

        if self.config["set_init"] == "safe":
            SafeSet(self.config).update_safe_set(X, self.aquisition)

        if not self.aquisition.has_safe_points(X):
            Logger.warn("Could not find safe set")

        return [next_param.detach().to("cpu"), reward.detach().to("cpu")]
