from gosafeopt.aquisitions.go_safe_opt import GoSafeOptState
import torch
from scipy.stats import norm

from gosafeopt.tools.logger import Logger


class Backup:

    def __init__(self, config, data, state_dict=None):
        self.data = data
        self.config = config
        self.state_dict = state_dict
        self.reset()
        self.goState = GoSafeOptState(config)

        self.idealTrajectory = None
        self.lastBackupFromInterior = False

    def isSafe(self, state):
        if self.goState.skipBackupAtRollout():
            return True

        backup = self.data.backup
        if self.idealTrajectory is not None:
            backup = torch.vstack([backup, self.idealTrajectory])

        diff = torch.linalg.norm(self.data.backup - state, axis=1)
        # diff /= diff.std()

        interior_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.config["safe_opt_interior_lb"]
        marginal_points = torch.min(self.data.backup_loss[:, 1:], axis=1)[0] > self.config["safe_opt_marginal_lb"]
        marginal_points = torch.logical_and(marginal_points, ~interior_points)

        diffInt = diff[interior_points]
        diffMarginal = diff[marginal_points]

        distribution = torch.distributions.Normal(0, self.config["safe_opt_sigma"])

        probsInterior = 1-2*(distribution.cdf(diffInt)-0.5)
        probsMarginal = 1-2*(distribution.cdf(diffMarginal)-0.5)

        # try:
        #     print(f"Marginal: {probsMarginal.max()}")
        #     print(f"Interior: {probsInterior.max()}")
        # except Exception:
        #     pass
        if torch.any(probsInterior > self.config["safe_opt_interior_prob"]):
            # print(f"Interior: {probsInterior.max()}")
            self.lastBackupFromInterior = True
            return True
        elif torch.any(probsMarginal > self.config["safe_opt_marginal_prob"]):
            # print(f"Marginal: {probsMarginal.max()}")
            self.lastBackupFromInterior = False 
            return True
        else:
            return False

    def getBackupPolicy(self, state):
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
        if self.data.failed_k is not None and self.goState.skipBackupAtRollout():
            mask = torch.ones(len(self.data.failed_x_rollout), dtype=bool)
            for i in range(len(self.data.failed_x_rollout)):
                mask[i] = self.isSafe(self.data.failed_x_rollout[i])

            s = len(self.data.failed_k)
            self.data.failed_k = self.data.failed_k[mask]
            self.data.failed_x_rollout = self.data.failed_x_rollout[mask]
            if s > len(self.data.failed_k):
                Logger.info(f"Removed {s-len(self.data.failed_k)}")

    def setIdealTrajectory(self, idealTrajectory):
        self.idealTrajectory = idealTrajectory

    def adddFail(self, k, state):
        self.data.append_failed(k, state)
