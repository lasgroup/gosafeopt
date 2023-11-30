from gosafeopt.aquisitions.go_safe_opt import GoSafeOpt, GoSafeOptState, OptimizationStep
import torch
from scipy.stats import norm
import numpy as np
from gosafeopt.tools.logger import Logger
from abc import ABC, abstractmethod
from torch import Tensor
from gosafeopt.tools.data import Data
import gosafeopt


class BackupStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def is_safe(self, state: Tensor) -> bool:
        pass

    @abstractmethod
    def get_backup_policy(self, state: Tensor) -> Tensor:
        pass

    @abstractmethod
    def after_rollout(self, param: Tensor, rewards: Tensor):
        pass

    def reset(self):
        pass


class GoSafeOptBackup(BackupStrategy):
    def __init__(
        self, data: Data, interior_lb: float, interior_prob: float, marginal_prob: float, marginal_lb: float, std: float
    ):
        super().__init__()
        self.interior_lb = interior_lb
        self.interior_prob = interior_prob
        self.marginal_prob = marginal_prob
        self.marginal_lb = marginal_lb
        self.std = std
        self.data = data
        self.lastBackupFromInterior = False

        self.backup_triggered = False

        self.reset()

    def after_rollout(self, param: Tensor, rewards: Tensor):
        if (
            GoSafeOpt.get_exploration_phase() == OptimizationStep.GLOBAL
            and not self.backup_triggered
            and not np.any(rewards[1:] < 0)
        ):
            param.to(gosafeopt.device)
            GoSafeOpt.go_to_local_exploration()
            SafeSet(self.c).add_new_safe_set(param.reshape(1, -1))
            SafeSet(self.c).change_to_latest_safe_set()

    def is_safe(self, state) -> bool:
        if self.data.backup is None or GoSafeOpt.get_exploration_phase() == OptimizationStep.LOCAL:
            return True

        diff = torch.linalg.norm(self.data.backup - state, axis=1)
        # diff /= diff.std()

        interior_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.interior_lb
        marginal_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.marginal_lb
        marginal_points = torch.logical_and(marginal_points, ~interior_points)

        diffInt = diff[interior_points]
        diffMarginal = diff[marginal_points]

        distribution = torch.distributions.Normal(0, self.std)

        probsInterior = 1 - 2 * (distribution.cdf(diffInt) - 0.5)
        probsMarginal = 1 - 2 * (distribution.cdf(diffMarginal) - 0.5)

        if torch.any(probsInterior > self.interior_prob):
            self.lastBackupFromInterior = True
            return True
        elif torch.any(probsMarginal > self.marginal_prob):
            self.lastBackupFromInterior = False
            return True
        else:
            self.backup_triggered = True
            return False

    def get_backup_policy(self, state):
        diff = torch.linalg.norm(self.data.backup - state, axis=1)

        interior_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.config["safe_opt_interior_lb"]
        marginal_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.config["safe_opt_marginal_lb"]
        marginal_points = torch.logical_and(marginal_points, ~interior_points)

        if self.lastBackupFromInterior:
            diff[~interior_points] = 1e10
        else:
            diff[~marginal_points] = 1e10

        Logger.info(f"Backup got backup with d: {torch.min(diff)} and Interiorset: {self.lastBackupFromInterior}")
        return self.data.backup_k[torch.argmin(diff)]

    def reset(self):
        self.backup_triggered = False
        if self.data.failed_k is not None and GoSafeOpt.get_exploration_phase == OptimizationStep.LOCAL:
            mask = torch.ones(len(self.data.failed_x_rollout), dtype=bool)
            for i in range(len(self.data.failed_x_rollout)):
                mask[i] = self.is_safe(self.data.failed_x_rollout[i])

            s = len(self.data.failed_k)
            self.data.failed_k = self.data.failed_k[mask]
            self.data.failed_x_rollout = self.data.failed_x_rollout[mask]
            if s > len(self.data.failed_k):
                Logger.info(f"Removed {s-len(self.data.failed_k)}")
