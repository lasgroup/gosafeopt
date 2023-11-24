from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
from rich.progress import track
import torch

from gosafeopt.optim import get_optimizer
from gosafeopt.aquisitions import get_aquisition
from gosafeopt.optim.base_optimizer import SafeSet
from gosafeopt.tools.data import Data
from botorch import fit_gpytorch_mll
from gosafeopt.tools.file import createFolderINE

from gosafeopt.tools.logger import Logger


import wandb


class Trainer:
    def __init__(self, config, context=None, state_dict=None, logger=None, data=None):
        self.config = config

        self.logger = logger

        self.data = Data() if data is None else data
        self.context = context
        self.state_dict = state_dict
        # TODO: remove this
        createFolderINE("{}/res".format(wandb.run.dir))
        createFolderINE("{}/video".format(wandb.run.dir))
        createFolderINE("{}/plot".format(wandb.run.dir))
        self.rewardMax = -1e10 * np.ones(self.config["dim_obs"])  # Something small
        self.bestK = None

    def train(self, experiment, model, safePoints: torch.Tensor):
        k = np.zeros(self.config["dim_params"])

        forExpression = (
            track(range(0, self.config["n_opt_samples"]), description="Training...")
            if self.config["show_progress"]
            else range(0, self.config["n_opt_samples"])
        )

        for i in forExpression:
            if safePoints is not None and i < safePoints.shape[0]:
                [k, acf_val] = [safePoints[i], torch.tensor([0])]
                reward, trajectory, backup_triggered, info = experiment.rollout(k, i)
                self.data.append_data(k.reshape(1, -1), reward.reshape(1, -1))
                gp = model(self.config, self.data, self.state_dict)
                aquisition = get_aquisition(gp, self.config, self.context, self.data)
            else:
                gp = model(self.config, self.data, self.state_dict)
                aquisition = get_aquisition(gp, self.config, self.context, self.data)
                optimizer = get_optimizer(aquisition, self.config, self.context)

                if (self.config["refit_interval"] != 0 and i % self.config["refit_interval"] == 0) and (
                    safePoints is None or i >= safePoints.shape[0]
                ):
                    mll = ExactMarginalLogLikelihood(gp.models[0].likelihood, gp.models[0])
                    Logger.info("Fitting GP")
                    fit_gpytorch_mll(mll)
                    for m in gp.models:
                        Logger.info(f"New Lenghtscale: {m.covar_module.base_kernel.lengthscale}")
                    self.state_dict = gp.state_dict()

                k, acf_val = optimizer.next_params()
                Logger.info("{}/{} next k: {}".format(i, self.config["n_opt_samples"], k))

                reward, trajectory, backup_triggered, info = experiment.rollout(k, i)

                if not backup_triggered:
                    self.data.append_data(k.reshape(1, -1), reward.reshape(1, -1))

            if torch.any(reward[1:] < 0):
                Logger.warn("Constraint violated at iteration {} with {} at {}".format(i, reward, k))

            if not backup_triggered:
                if reward[0] > self.rewardMax[0]:
                    self.rewardMax = reward
                    self.bestK = k
                    experiment.env.best_param = k
                    Logger.success("New minimum at Iteration: {},yMin:{} at {}".format(i, self.rewardMax, k))
                else:
                    Logger.info("Reward at Iteration: {},y:{} at {}".format(i, reward, k))

                SafeSet(self.config).calculate_current_set(reward[0])

                if experiment.backup is not None:
                    self.data.append_backup(trajectory, reward, k)

            if self.logger is not None:
                self.logger.log(
                    gp, trajectory, k, reward, self.data, self.rewardMax, acf_val, backup_triggered, i, info
                )

            # Save progress
            if (i > 0 and not i % self.config["save_interval"]) or i == self.config["n_opt_samples"] - 1:
                self.data.save("{}/res".format(wandb.run.dir))
                torch.save(gp.state_dict(), "{}/res/model.pth".format(wandb.run.dir))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
