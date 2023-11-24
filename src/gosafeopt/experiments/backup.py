from gosafeopt.aquisitions.go_safe_opt import GoSafeOptState, OptimizationStep
import torch
from scipy.stats import norm
import numpy as np
from gosafeopt.tools.logger import Logger

# TODO: generic backup class


class Backup:
    def __init__(self, config, data, state_dict=None):
        self.data = data
        self.config = config
        self.state_dict = state_dict
        self.reset()
        self.go_state = GoSafeOptState(config)

        self.idealTrajectory = None
        self.lastBackupFromInterior = False

    def is_safe(self, state) -> bool:
        if self.data.backup is None or self.go_state.get_step() == OptimizationStep.LOCAL:
            return True

        diff = torch.linalg.norm(self.data.backup - state, axis=1)
        # diff /= diff.std()

        interior_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.config["safe_opt_interior_lb"]
        marginal_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.config["safe_opt_marginal_lb"]
        marginal_points = torch.logical_and(marginal_points, ~interior_points)

        diffInt = diff[interior_points]
        diffMarginal = diff[marginal_points]

        distribution = torch.distributions.Normal(0, self.config["safe_opt_sigma"])

        probsInterior = 1 - 2 * (distribution.cdf(diffInt) - 0.5)
        probsMarginal = 1 - 2 * (distribution.cdf(diffMarginal) - 0.5)

        if torch.any(probsInterior > self.config["safe_opt_interior_prob"]):
            self.lastBackupFromInterior = True
            return True
        elif torch.any(probsMarginal > self.config["safe_opt_marginal_prob"]):
            self.lastBackupFromInterior = False
            return True
        else:
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
        if self.data.failed_k is not None and self.go_state.get_step() == OptimizationStep.LOCAL:
            mask = torch.ones(len(self.data.failed_x_rollout), dtype=bool)
            for i in range(len(self.data.failed_x_rollout)):
                mask[i] = self.is_safe(self.data.failed_x_rollout[i])

            s = len(self.data.failed_k)
            self.data.failed_k = self.data.failed_k[mask]
            self.data.failed_x_rollout = self.data.failed_x_rollout[mask]
            if s > len(self.data.failed_k):
                Logger.info(f"Removed {s-len(self.data.failed_k)}")

    def add_fail(self, param, observation):
        self.data.append_failed(param, torch.tensor(observation))
