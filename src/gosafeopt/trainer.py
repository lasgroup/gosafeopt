from botorch.models import ModelListGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
from rich.progress import track
import torch
from gosafeopt.aquisitions.base_aquisition import BaseAquisition
from gosafeopt.experiments.experiment import Experiment

from gosafeopt.optim.base_optimizer import BaseOptimizer, SafeSet
from gosafeopt.tools.data import Data
from botorch import fit_gpytorch_mll
from gosafeopt.tools.file import createFolderINE
from gosafeopt.models.model import ModelGenerator
from gosafeopt.tools.logger import Logger
from torch import Tensor
from typing import Optional
from gosafeopt.tools.data_logger import WandbLogger
import wandb


class Trainer:
    def __init__(
        self,
        dim_params: int,
        dim_obs: int,
        n_opt_samples: int,
        save_interval: int = 5,
        show_progress: bool = False,
        refit_interval: int = 0,
        context: Optional[Tensor] = None,
        logger: Optional[WandbLogger] = None,
        data: Optional[Data] = None,
    ):
        self.dim_params = dim_params
        self.dim_obs = dim_obs
        self.n_opt_samples = n_opt_samples
        self.show_progress = show_progress
        self.save_interval = save_interval
        self.refit_interval = refit_interval

        self.logger = logger
        self.data = Data() if data is None else data
        self.context = context
        self.rewardMax = -1e10 * np.ones(self.dim_obs)  # Something small
        self.bestK = None

        # Create data folders if not present
        createFolderINE("{}/res".format(wandb.run.dir))
        createFolderINE("{}/video".format(wandb.run.dir))
        createFolderINE("{}/plot".format(wandb.run.dir))

    # TODO: make parameters explicit instead of config
    def train(
        self,
        experiment: Experiment,
        model: ModelGenerator,
        optimizer: BaseOptimizer,
        aquisition: BaseAquisition,
        safePoints: torch.Tensor,
    ):
        k = np.zeros(self.dim_params)

        forExpression = (
            track(range(0, self.n_opt_samples), description="Training...")
            if self.show_progress
            else range(0, self.n_opt_samples)
        )

        for episode in forExpression:
            # Collect data from known safe points
            if safePoints is not None and episode < safePoints.shape[0]:
                [k, acf_val] = [safePoints[episode], torch.tensor([0])]
                reward, trajectory, backup_triggered, info = experiment.rollout(k, episode)
                self.data.append_data(k.reshape(1, -1), reward.reshape(1, -1))
            # Optimization loop
            else:
                if episode == safePoints.shape[0]:
                    # TODO: this is not so nice...
                    model = model.generate(self.data)  # We need a generator since we have no data previously...

                aquisition.update_model(model)
                aquisition.before_optimization()
                self.refit_reward_model(model, episode)

                k, acf_val = optimizer.next_params()
                Logger.info("{}/{} next k: {}".format(episode, self.n_opt_samples, k))

                reward, trajectory, backup_triggered, info = experiment.rollout(k, episode)

                aquisition.after_optimization()

                if not backup_triggered:
                    self.data.append_data(k.reshape(1, -1), reward.reshape(1, -1))
                    model = model.condition_on_observations(
                        [k.reshape(1, 1, -1), k.reshape(1, 1, -1)], reward.reshape(1, 1, -1)
                    )

            if torch.any(reward[1:] < 0):
                Logger.warn("Constraint violated at iteration {} with {} at {}".format(episode, reward, k))

            if not backup_triggered:
                if reward[0] > self.rewardMax[0]:
                    self.rewardMax = reward
                    self.bestK = k
                    experiment.env.best_param = k
                    Logger.success("New minimum at Iteration: {},yMin:{} at {}".format(episode, self.rewardMax, k))
                else:
                    Logger.info("Reward at Iteration: {},y:{} at {}".format(episode, reward, k))

                SafeSet.calculate_current_set(reward[0])

                if experiment.backup is not None:
                    self.data.append_backup(trajectory, reward, k)

            if self.logger is not None:
                self.logger.log(
                    model, trajectory, k, reward, self.data, self.rewardMax, acf_val, backup_triggered, episode, info
                )

            # Save progress
            if (episode > 0 and not episode % self.save_interval) or episode == self.n_opt_samples - 1:
                self.data.save("{}/res".format(wandb.run.dir))
                torch.save(model.state_dict(), "{}/res/model.pth".format(wandb.run.dir))

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def refit_reward_model(self, model: ModelListGP, episode: int):
        if self.refit_interval != 0 and episode % self.refit_interval == 0:
            # Only fit reward model
            mll = ExactMarginalLogLikelihood(model.models[0].likelihood, model.models[0])
            Logger.info("Fitting GP")
            fit_gpytorch_mll(mll)
            Logger.info(f"New Lenghtscale: {model.models[0].covar_module.base_kernel.lengthscale}")
            self.state_dict = model.state_dict()
