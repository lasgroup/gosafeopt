from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import torch
from numpy.typing import NDArray

import gosafeopt
from gosafeopt.aquisitions.go_safe_opt import GoSafeOpt, OptimizationStep
from gosafeopt.optim.safe_set import SafeSet
from gosafeopt.tools.data import Data
from gosafeopt.tools.logger import Logger


class BackupStrategy(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def is_safe(self, state: NDArray, reward: NDArray) -> bool:
        pass

    @abstractmethod
    def get_backup_policy(self, state: NDArray) -> NDArray:
        pass

    @abstractmethod
    def after_rollout(self, param: NDArray, rewards: NDArray, backup_triggered: bool):
        pass

    @abstractmethod
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

        self.reset()

    def after_rollout(self, param: NDArray, rewards: NDArray, backup_triggerd: bool):
        if (
            GoSafeOpt.get_exploration_phase() == OptimizationStep.GLOBAL
            and not backup_triggerd
            and not np.any(rewards[1:] < 0)
        ):
            p = torch.from_numpy(param)
            p.to(gosafeopt.device)
            GoSafeOpt.go_to_local_exploration()
            SafeSet.add_new_safe_set(p.reshape(1, -1))
            SafeSet.change_to_latest_safe_set()

    def is_safe(self, state: NDArray, reward: Optional[NDArray] = None) -> bool:
        if self.data.backup is None or GoSafeOpt.get_exploration_phase() == OptimizationStep.LOCAL:
            return True
        elif reward is not None and np.any(reward[1:] < 0):
            return False

        if self.data.backup_loss is None:
            raise Exception("Backup loss is empty")

        diff = torch.linalg.norm(self.data.backup - state, axis=1)
        # diff /= diff.std()  # noqa: ERA001

        interior_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.interior_lb  # type: ignore
        marginal_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.marginal_lb  # type: ignore

        marginal_points = torch.logical_and(marginal_points, ~interior_points)

        diff_int = diff[interior_points]
        diff_marginal = diff[marginal_points]

        distribution = torch.distributions.Normal(0, self.std)

        probs_interior = 1 - 2 * (distribution.cdf(diff_int) - 0.5)
        probs_marginal = 1 - 2 * (distribution.cdf(diff_marginal) - 0.5)

        if torch.any(probs_interior > self.interior_prob):
            self.lastBackupFromInterior = True
            return True
        elif torch.any(probs_marginal > self.marginal_prob):
            self.lastBackupFromInterior = False
            return True
        else:
            return False

    def get_backup_policy(self, state: NDArray):
        if self.data.backup_k is None:
            raise Exception("Backup is none")

        diff = torch.linalg.norm(self.data.backup - state, axis=1)

        interior_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.interior_lb  # type: ignore
        marginal_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.marginal_lb  # type: ignore
        marginal_points = torch.logical_and(marginal_points, ~interior_points)

        if self.lastBackupFromInterior:
            diff[~interior_points] = 1e10
        else:
            diff[~marginal_points] = 1e10

        Logger.info(f"Backup got backup with d: {torch.min(diff)} and Interiorset: {self.lastBackupFromInterior}")
        return self.data.backup_k[torch.argmin(diff)]

    def reset(self):
        if (
            self.data.failed_k is not None
            and self.data.failed_x_rollout is not None
            and GoSafeOpt.get_exploration_phase == OptimizationStep.LOCAL
        ):
            mask = np.ones(len(self.data.failed_x_rollout), dtype=bool)
            for i in range(len(self.data.failed_x_rollout)):
                mask[i] = self.is_safe(self.data.failed_x_rollout[i].numpy())

            s = len(self.data.failed_k)
            self.data.failed_k = self.data.failed_k[mask]
            self.data.failed_x_rollout = self.data.failed_x_rollout[mask]
            if s > len(self.data.failed_k):
                Logger.info(f"Removed {s-len(self.data.failed_k)}")
